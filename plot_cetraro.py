import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import svgutils.transform as sg

from config import data_path
from multiarea_model import MultiAreaModel
from figures.Schmidt2018_dyn.plotcolors import myred, myblue
from figures.Schmidt2018_dyn.helpers import population_labels


area = 'V1'
net_label = 'ad291b6d46e107e0a12480df884941d0'
sim_labels = {
    'small': '1abd7150fb1a6f41a5dfe1ec16b2a387',
    'medium': '1275b068b789e10edf5fd15928a07036',
    'big': 'cb7c3d87f6c2da1ccb8e76c321ef3dbe'
}

t_min = 1000.
t_max = 2000.

icolor = myred
ecolor = myblue

frac_neurons = 0.03

M = MultiAreaModel(net_label)
populations = M.structure[area]


# spike data
spike_data = {}
for name, label in sim_labels.items():
    spike_data[name] = {}
    for pop in populations:
        spike_file = '{}-spikes-{}-{}.npy'.format(label, area, pop)
        spike_file = os.path.join(data_path, label, 'recordings', spike_file)
        spike_data[name][pop] = np.load(spike_file)
    print('loaded {}'.format(name))


fig = plt.figure(figsize=(13, 6))
axes = {}

gs1 = gridspec.GridSpec(1, 3)
gs1.update(left=0.06, right=0.96, top=0.50, wspace=0.7, bottom=0.10)
axes['A'] = plt.subplot(gs1[0, 0])
axes['B'] = plt.subplot(gs1[0, 1])
axes['C'] = plt.subplot(gs1[0, 2])
axes_names = {'A': 'small', 'B': 'medium', 'C': 'big'}

for label, ax in axes.items():
    label_pos = [-0.2, 2.1]
    plt.text(label_pos[0], label_pos[1], label,
             fontdict={'fontsize': 18, 'weight': 'bold',
                       'horizontalalignment': 'left', 'verticalalignment':
                       'bottom'}, transform=ax.transAxes)
    label_pos = [0.0, 1.06]
    plt.text(label_pos[0], label_pos[1], area,
             fontdict={'fontsize': 14, 'weight': 'normal',
                       'horizontalalignment': 'right', 'verticalalignment':
                       'bottom'}, transform=ax.transAxes)

for label, ax in axes.items():
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.yaxis.set_ticks_position("none")
    ax.xaxis.set_ticks_position("bottom")

    name = axes_names[label]
    if name in spike_data:
        n_pops = len(spike_data[name])
        # Determine number of neurons that will be plotted for this area (for
        # vertical offset)
        offset = 0
        n_to_plot = {}
        for pop in populations:
            n_to_plot[pop] = int(M.N[area][pop] * frac_neurons)
            offset = offset + n_to_plot[pop]
        y_max = offset + 1
        prev_pop = ''
        yticks = []
        yticklocs = []
        for jj, pop in enumerate(M.structure[area]):
            print('plotting {} {}'.format(name, pop))
            if pop[0:-1] != prev_pop:
                prev_pop = pop[0:-1]
                yticks.append('L' + population_labels[jj][0:-1])
                yticklocs.append(offset - 0.5 * n_to_plot[pop])
            ind = np.where(np.logical_and(
                spike_data[name][pop][:, 1] <= t_max,
                spike_data[name][pop][:, 1] >= t_min
            ))
            pop_data = spike_data[name][pop][ind]
            pop_neurons = np.unique(pop_data[:, 0])
            neurons_to_ = np.arange(
                np.min(spike_data[name][pop][:, 0]),
                np.min(spike_data[name][pop][:, 0]) + n_to_plot[pop],
                1
            )

            if pop.find('E') > (-1):
                pcolor = ecolor
            else:
                pcolor = icolor

            for kk in range(n_to_plot[pop]):
                spike_times = pop_data[pop_data[:, 0] == neurons_to_[kk], 1]

                _ = ax.plot(spike_times, np.zeros(len(spike_times)) +
                            offset - kk, '.', color=pcolor, markersize=1)
            offset = offset - n_to_plot[pop]
        y_min = offset
        ax.set_xlim([t_min, t_max])
        ax.set_ylim([y_min, y_max])
        ax.set_yticklabels(yticks)
        ax.set_yticks(yticklocs)
        ax.set_xlabel('Time (s)', labelpad=-0.1)
        ax.set_xticks([t_min, (t_min + t_max)/2, t_max])
        ax.set_xticklabels([r'$1.$', r'$1.5$', r'$2.$'])


print('doing svg magic')
svgMpl = sg.from_mpl(fig, savefig_kw=dict(transparent=True))
svgSketchSmall = sg.fromfile('conn_modules_small.svg')
svgSketchMedium = sg.fromfile('conn_modules_medium.svg')
svgSketchBig = sg.fromfile('conn_modules_big.svg')
svgPlotMpl = svgMpl.getroot()
svgPlotSketchSmall = svgSketchSmall.getroot()
svgPlotSketchMedium = svgSketchMedium.getroot()
svgPlotSketchBig = svgSketchBig.getroot()
svgPlotSketchSmall.moveto(56, 10, scale=0.311)
svgPlotSketchMedium.moveto(381, 10, scale=0.311)
svgPlotSketchBig.moveto(706, 10, scale=0.311)
xSize, ySize = svgMpl.get_size()
xSize = float(xSize)
ySize = float(ySize)
svgFig = sg.SVGFigure(xSize, ySize)
svgFig.append([
    svgPlotMpl, svgPlotSketchSmall, svgPlotSketchMedium, svgPlotSketchBig
])
svgFig.save('cetraro.svg')

# inkscape --export-pdf=cetraro.pdf cetraro.svg
