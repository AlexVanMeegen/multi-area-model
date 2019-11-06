from pprint import pprint
from multiarea_model import MultiAreaModel

network_label = 'ad291b6d46e107e0a12480df884941d0'
# ["V1"]
# simulation_label = '1abd7150fb1a6f41a5dfe1ec16b2a387'
# ["V1", "V2", "V4", "AITv", "PITv", "AITd", "CITv", "TF", "CITd", "TH"]
# simulation_label = '1275b068b789e10edf5fd15928a07036'
# full MAM
simulation_label = 'cb7c3d87f6c2da1ccb8e76c321ef3dbe'

area = 'V1'
t_min = 3000.0
t_max = 3500.0


M = MultiAreaModel(
    network_label,
    simulation=True,
    sim_spec=simulation_label,
    analysis=True
)

M.analysis.create_pop_rates(
    t_min=t_min,
    t_max=t_max
)
M.analysis.save()
pprint(M.analysis.pop_rates[area])

M.analysis.single_dot_display(
    area=area,
    frac_neurons=0.03,
    t_min=t_min,
    t_max=t_max,
    output='pdf'
)
