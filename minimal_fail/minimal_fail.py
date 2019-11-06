import nest
import numpy as np


dt = 0.1
T = 50.0

rate_ext = 10.0
K = np.load('K.npy')
J = np.load('J.npy')
dim = np.shape(K)[0]

NP = {
    'theta': 15.0, 'V_reset': 0.0, 'tau_m': 10.0, 'tau_syn': 0.5, 't_ref': 2.0,
    'tau': 1.0
}
tau = NP['tau_m'] * 1e-3

nest.ResetKernel()
nest.set_verbosity('M_ERROR')
nest.SetKernelStatus({
    'resolution': dt, 'use_wfr': False, 'print_time': False,
    'overwrite_files': True
})

# create neurons for external drive
drive = nest.Create('siegert_neuron', 1, params={
    'rate': rate_ext, 'mean': rate_ext
})

# create neurons representing populations
neurons = nest.Create('siegert_neuron', dim, params=NP)

# external drive
syn_dict = {
    'drift_factor': tau * np.array([K[:, -1] * J[:, -1]]).transpose(),
    'diffusion_factor': tau * np.array([K[:, -1] * J[:, -1]**2]).transpose(),
    'model': 'diffusion_connection',
    'receptor_type': 0
}
nest.Connect(drive, neurons, 'all_to_all', syn_dict)

# external DC drive (expressed in mV)
DC_drive = nest.Create('siegert_neuron', 1, params={
    'rate': 1., 'mean': 1.
})
syn_dict = {
    'drift_factor': 0.,
    'diffusion_factor': 0.,
    'model': 'diffusion_connection',
    'receptor_type': 0
}
nest.Connect(DC_drive, neurons, 'all_to_all', syn_dict)

# network connections
syn_dict = {
    'drift_factor': tau * K[:, :-1] * J[:, :-1],
    'diffusion_factor': tau * K[:, :-1] * J[:, :-1]**2,
    'model': 'diffusion_connection',
    'receptor_type': 0
}
nest.Connect(neurons, neurons, 'all_to_all', syn_dict)

# Set initial rates of neurons:
nest.SetStatus(neurons, {'rate': 0.})

# create recording device
multimeter = nest.Create('multimeter', params={
    'record_from': ['rate'], 'interval': 1., 'to_screen': False,
    'to_file': False, 'to_memory': True
})
nest.Connect(multimeter, neurons)
nest.Connect(multimeter, drive)

# simulate
nest.Simulate(T)

data = nest.GetStatus(multimeter)[0]['events']
r = np.array([
    np.insert(
        data['rate'][np.where(data['senders'] == n)],
        0,
        0.)
    for ii, n in enumerate(neurons)
])

r_mf = np.load('mf_rates.npy')
print(r[:8, -1])
print(r_mf[:8, -1])
assert(np.allclose(r[:, -1], r_mf[:, -1]))
