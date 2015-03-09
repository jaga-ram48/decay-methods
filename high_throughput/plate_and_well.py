from methods.HTExperiment import Experiment
from methods.PlotOptions import PlotOptions
PlotOptions(uselatex=True)
import matplotlib.pylab as plt

zhang = Experiment('zhang').split()

# zhang.plate_to_plate_variation()




import numpy as np
from methods.PlotOptions import lighten_color, solarized, layout_pad

# Add plate and well information to the raw fits
annotated_fits = zhang.raw_blfit.join(
    zhang.raw_names.loc[:, ['plate', 'well']])

well_breakdown = annotated_fits.well.str.extract(
    '(?P<well_row>[A-Za-z]{1})(?P<well_col>\d{1,2})')
annotated_fits = annotated_fits.join(well_breakdown).drop(['well'], 1)

multi_index = annotated_fits.set_index(['plate', 'well_col', 'well_row'])

def line_box(ax, group, param, color, t=False):
    pivot = multi_index.loc[:, param].unstack(group)
    if t: pivot = pivot.T
    
    lower = pivot.quantile(0.25)
    upper = pivot.quantile(0.75)
    for i, (l,u) in enumerate(zip(lower, upper)):
        ax.plot([i+1, i+1], [l, u],
                color=lighten_color(color, 0.75), zorder=0)
    ax.plot(np.arange(pivot.shape[1]) + 1,
            pivot.median().values, '.', color=color, zorder=1)


    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.set_ylabel(param)



# Plate-to-plate variation
fig, axmatrix = plt.subplots(nrows=5, ncols=1, sharex=True)

params = zhang.blfit.columns.values
for ax, param, color in zip(axmatrix, params,
                            solarized.values()):
    line_box(ax, 'plate', param, color)

axmatrix[0].set_ylim([0.75, 1.0])
axmatrix[1].set_ylim([4, 5.5])
axmatrix[2].set_ylim([0.01, 0.04])
axmatrix[3].set_ylim([22.5, 28])
axmatrix[4].set_ylim([-np.pi, np.pi])
axmatrix[4].set_yticks([-np.pi, 0, np.pi])
axmatrix[4].set_yticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
axmatrix[4].set_xlim([0, zhang.num_plates+1])
axmatrix[4].set_xlabel('Plate')
fig.tight_layout(**layout_pad)


# Well-to-well variation
fig, axmatrix = plt.subplots(nrows=5, ncols=1, sharex=True)

params = zhang.blfit.columns.values
for ax, param, color in zip(axmatrix, params,
                            solarized.values()):
    line_box(ax, 'plate', param, color, t=True)

axmatrix[0].set_ylim([0.75, 1.0])
axmatrix[1].set_ylim([4, 5.5])
axmatrix[2].set_ylim([0.01, 0.04])
axmatrix[3].set_ylim([10, 50])
axmatrix[4].set_ylim([-np.pi, np.pi])
axmatrix[4].set_yticks([-np.pi, 0, np.pi])
axmatrix[4].set_yticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
axmatrix[4].set_xlim([0, 385])
axmatrix[4].set_xlabel('Well')
fig.tight_layout(**layout_pad)






plt.show()
