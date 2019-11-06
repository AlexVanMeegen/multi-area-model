import numpy as np
import os

from multiarea_model import MultiAreaModel
from start_jobs import start_job
from config import submit_cmd, jobscript_template
from config import base_path
from figures.Schmidt2018_dyn.network_simulations import NEW_SIM_PARAMS


network_params, sim_params = NEW_SIM_PARAMS['Fig5'][0]

network_params['connection_params']['K_stable'] = os.path.join(
    base_path, 'figures/SchueckerSchmidt2017/K_prime_original.npy'
)
network_params['connection_params']['replace_non_simulated_areas'] = 'het_poisson_stat'
network_params['connection_params']['replace_cc_input_source'] = os.path.join(
    base_path, 'tests/fullscale_rates.json'
)

sim_params['num_processes'] = 10
sim_params['areas_simulated'] = ['V1', 'V2', 'V4', 'AITv', 'PITv', 'AITd', 'CITv', 'TF', 'CITd', 'TH']

theory_params = {'dt': 0.1}

M = MultiAreaModel(network_params, simulation=True,
                   sim_spec=sim_params,
                   theory=True,
                   theory_spec=theory_params)
p, r = M.theory.integrate_siegert()
print("Mean-field theory predicts an average "
      "rate of {0:.3f} spikes/s across all populations.".format(np.mean(r[:, -1])))
start_job(M.simulation.label, submit_cmd, jobscript_template)
