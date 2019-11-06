import os
import numpy as np
from pprint import pprint

from multiarea_model import MultiAreaModel
from config import base_path

for target_neuron_number in np.arange(800, 4000, 200):
    print("Target neuron number: {}".format(target_neuron_number))
    scaling_factor = target_neuron_number / 197935.95927204541
    print("Scaling factor: {}".format(scaling_factor))

    d = {}
    conn_params = {'replace_non_simulated_areas': 'het_poisson_stat',
                   'replace_cc_input_source': os.path.join(base_path, 'tests/fullscale_rates.json'),
                   'g': -11.,
                   'K_stable': 'K_stable.npy',
                   'fac_nu_ext_TH': 1.2,
                   'fac_nu_ext_5E': 1.125,
                   'fac_nu_ext_6E': 1.41666667,
                   'av_indegree_V1': 3950.}
    input_params = {'rate_ext': 10.25}
    neuron_params = {'V0_mean': -150.,
                     'V0_sd': 50.}
    network_params = {'N_scaling': scaling_factor,
                      'K_scaling': scaling_factor,
                      'fullscale_rates': os.path.join(base_path, 'tests/fullscale_rates.json'),
                      'input_params': input_params,
                      'connection_params': conn_params,
                      'neuron_params': neuron_params}

    sim_params = {'t_sim': 2000.,
                  'num_processes': 1,
                  'local_num_threads': 1,
                  'areas_simulated': ['V1'],
                  'recording_dict': {'record_vm': False}}

    theory_params = {'dt': 0.1}

    M = MultiAreaModel(network_params, simulation=True,
                       sim_spec=sim_params,
                       theory=True,
                       theory_spec=theory_params)

    nu_ext = M.params['input_params']['rate_ext']
    tau_m = M.params['neuron_params']['single_neuron_dict']['tau_m']
    tau_syn = M.params['neuron_params']['single_neuron_dict']['tau_syn_ex']
    C_m = M.params['neuron_params']['single_neuron_dict']['C_m']
    conn_prob_thal = np.array([0.0, 0.0, 0.0983, 0.0619, 0.0, 0.0, 0.0512, 0.0196])

    neurons = np.round(M.N_vec[:8]).astype(np.int32)
    frac_exc = neurons[::2].sum() / neurons.sum()
    K_rec = np.round(M.K_matrix[:8, :8]).astype(np.int32)
    K_out = np.dot(M.K_matrix[8:, :8].T, M.N_vec[8:])
    K_out /= K_out.sum()
    K_V2 = np.round(M.K_matrix[:8, 8:16].sum(axis=1)).astype(np.int32)
    K_ctx = np.round(M.K_matrix[:8, 8:].sum(axis=1)).astype(np.int32)
    K_th = np.round(neurons * conn_prob_thal).astype(np.int32)
    J_rec = M.J_matrix[:8, :8]
    mu_ext = tau_m * M.add_DC_drive[:8] / C_m + 1e-3 * tau_m * (M.J_matrix * M.K_matrix * nu_ext)[:8, -1]

    pprint('Neuron number')
    pprint(neurons)
    pprint('Fraction E neurons')
    pprint(frac_exc)
    pprint('Indegree recurrent')
    pprint(K_rec)
    pprint('External DC drive')
    pprint(mu_ext)
    pprint('Outdegree')
    pprint(K_out)
    pprint('Indegree from thalamus')
    pprint(K_th)
    pprint('Indegree from cortex')
    pprint(K_ctx)
    pprint('Indegree from V2')
    pprint(K_V2)
    pprint('Weights recurrent')
    pprint(J_rec)
    pprint('Average connection prob')
    pprint(M.K_matrix[:8, :8].sum().sum() / M.N_vec[:8].sum())
    np.savez(
        'v1_{}.npz'.format(target_neuron_number),
        population_sizes=neurons, indegrees_recurrent=K_rec,
        outdegrees=K_out, indegrees_thalamus=K_th, indegrees_cortex=K_ctx,
        weights_recurrent=J_rec, proportion_excitatory=frac_exc,
        dc_drive=mu_ext
    )

    p, r = M.theory.integrate_siegert()
    print("Mean-field theory predicts an average "
          "rate of {0:.3f} spikes/s across all populations.".format(np.mean(r[:, -1])))
    M.simulation.simulate()

    M = MultiAreaModel(network_params, simulation=True,
                       sim_spec=sim_params,
                       theory=True,
                       theory_spec=theory_params,
                       analysis=True)

    t_min = 1500.0
    t_max = 2000.0
    M.analysis.create_pop_rates(
        t_min=t_min,
        t_max=t_max
    )
    M.analysis.save()
    pprint(M.analysis.pop_rates['V1'])

    M.analysis.single_dot_display(
        area='V1',
        frac_neurons=1.0,
        t_min=t_min,
        t_max=t_max,
        output='png'
    )
