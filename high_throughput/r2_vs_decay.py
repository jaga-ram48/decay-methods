import pandas as pd
import numpy as np

from methods.HTExperiment import Experiment

zhang = Experiment('zhang').split()
zhang.raw_lumin = zhang.raw_lumin.iloc[:,2:]



from methods.PlotOptions import PlotOptions, layout_pad
PlotOptions(uselatex=True)
import matplotlib.pyplot as plt


# Create a plot with several subplots
import matplotlib.gridspec as gridspec
gs_left = gridspec.GridSpec(1,1)
gs_right = gridspec.GridSpec(2,2)
gs_left.update(left=0.06, right=0.475, top=0.95, bottom=0.13)
gs_right.update(left=0.575, right=0.99, top=0.95, bottom=0.13,
                wspace=0.3)

fig = plt.figure(figsize=(6.5,2.5))


# Large block to hold the R2 vs Damping rate subplot
ax1 = fig.add_subplot(gs_left[0,0])

# Big subplot for common axis labels
ax_all = fig.add_subplot(gs_right[:,:])
ax_all.spines['top'].set_color('none')
ax_all.spines['bottom'].set_color('none')
ax_all.spines['left'].set_color('none')
ax_all.spines['right'].set_color('none')
ax_all.tick_params(labelcolor='w', top='off', bottom='off', left='off',
                   right='off')

# Smaller subplots to show example trajectories from each division
ax2 = fig.add_subplot(gs_right[0,0])
ax3 = fig.add_subplot(gs_right[1,0], sharex=ax2)
ax4 = fig.add_subplot(gs_right[0,1], sharex=ax2)
ax5 = fig.add_subplot(gs_right[1,1], sharex=ax2)
plt.setp(ax2.get_xticklabels(), visible=False)
# plt.setp(ax3.get_xticklabels(), visible=False)
plt.setp(ax4.get_xticklabels(), visible=False)
# plt.setp(ax5.get_xticklabels(), visible=False)


# Define the relevant ranges for damping and fit quality
high_r2 = zhang.raw_blfit.R2 >= 0.8
high_decay = zhang.raw_blfit.decay >= 0

# Plot each range with a different color
ax1.plot(zhang.raw_blfit.decay[high_r2 & high_decay],
         zhang.raw_blfit.R2[high_r2 & high_decay],
         '.', markersize=0.75, zorder=1, rasterized=True)
ax1.plot(zhang.raw_blfit.decay[high_r2 & ~high_decay],
         zhang.raw_blfit.R2[high_r2 & ~high_decay],
         '.', markersize=0.75, zorder=1, rasterized=True)
ax1.plot(zhang.raw_blfit.decay[~high_r2],
         zhang.raw_blfit.R2[~high_r2],
         '.', markersize=0.75, color='gray', zorder=1, rasterized=True)

# Add division lines
ax1.axhline(0.8, color='gray', linestyle='--')
ax1.axvline(0, color='gray', linestyle='--')

# Appropriate axis scaling
ax1.set_xlim([-0.025, 0.07])
ax1.set_ylim([0.5, 1.0])

# Add text showing the number of points in each quadrant
x_eps = -0.002
y_eps = 0.0075

n_hh = sum(high_r2 & high_decay)
n_lh = sum(~high_r2 & high_decay)
n_ll = sum(~high_r2 & ~high_decay)
n_hl = sum(high_r2 & ~high_decay)

ax1.text(0.07 + x_eps, 0.8 + y_eps, 
         '$n = %d$' % n_hh,
         horizontalalignment='right', verticalalignment='bottom')

ax1.text(0.07 + x_eps, 0.5 + y_eps,
         '$n = %d$' % n_lh,
         horizontalalignment='right', verticalalignment='bottom')

ax1.text(0.0 + x_eps, 0.5 + y_eps,
         '$n = %d$' % n_ll,
         horizontalalignment='right', verticalalignment='bottom')

ax1.text(0.0 + x_eps, 0.8 + y_eps,
         '$n = %d$' % n_hl,
         horizontalalignment='right', verticalalignment='bottom')

# Add appropriate axes labels
ax1.set_xlabel(r'Damping Rate $\left(\nicefrac{1}{\textrm{hrs}}\right)$')
ax1.set_ylabel(r'$R^2$')



# Plot example fits for trajectories in each quadrant
from methods.sinusoid_estimation import fit_data
def fit(row, names, outliers=False, full=True):
    """ Wrapper function of the exponential sinusoid fitting function """
    return fit_data(row.values, names.iloc[row.name].sampling_period,
                    outliers=outliers, full=full)


# Set a random seed
np.random.seed(20150117)

randint_hh = np.random.randint(0, n_hh)
randint_lh = np.random.randint(0, n_lh)
# randint_ll = np.random.randint(0, n_ll)
randint_hl = np.random.randint(0, n_hl)

fit1 = fit(
    zhang.raw_lumin[high_r2 & high_decay].iloc[randint_hh],
    zhang.raw_names)
fit2 = fit(
    zhang.raw_lumin[~high_r2 & high_decay].iloc[randint_lh],
    zhang.raw_names)
fit4 = fit(
    zhang.raw_lumin[high_r2 & ~high_decay].iloc[randint_hl],
    zhang.raw_names)

# This one sadly is usually off the plot. (ax1), so we restrict the
# sampling range
r2_range = (zhang.raw_blfit.R2 > 0.5) & (zhang.raw_blfit.R2 < 0.8)
decay_range = (zhang.raw_blfit.decay > -0.025) & (zhang.raw_blfit.decay < 0)
randint_ll = np.random.randint(0, sum(r2_range & decay_range))

fit3 = fit(
    zhang.raw_lumin[r2_range & decay_range].iloc[randint_ll],
    zhang.raw_names)

# Plot each fit on the correct axis
def plot_fit(ax, fit):
    t = np.arange(len(fit.x_hat))*fit.sampling_period
    ax.plot(t, fit.x_lowpass, '.')
    ax.plot(t, fit.x_hat, '-')
    

plot_fit(ax4, fit1)
plot_fit(ax5, fit2)
plot_fit(ax3, fit3)
plot_fit(ax2, fit4)



# Plot the locations of the fits on the main chart
a_loc = (zhang.raw_blfit[high_r2 & high_decay].iloc[randint_hh].decay,
         zhang.raw_blfit[high_r2 & high_decay].iloc[randint_hh].R2)
b_loc = (zhang.raw_blfit[~high_r2 & high_decay].iloc[randint_lh].decay,
         zhang.raw_blfit[~high_r2 & high_decay].iloc[randint_lh].R2)
c_loc = (zhang.raw_blfit[r2_range & decay_range].iloc[randint_ll].decay,
         zhang.raw_blfit[r2_range & decay_range].iloc[randint_ll].R2)
d_loc = (zhang.raw_blfit[high_r2 & ~high_decay].iloc[randint_hl].decay,
         zhang.raw_blfit[high_r2 & ~high_decay].iloc[randint_hl].R2)

ax1.plot(a_loc[0], a_loc[1], 'go', markeredgecolor='white',
         markeredgewidth=0.5)
ax1.plot(b_loc[0], b_loc[1], 'go', markeredgecolor='white',
         markeredgewidth=0.5)
ax1.plot(c_loc[0], c_loc[1], 'go', markeredgecolor='white',
         markeredgewidth=0.5)
ax1.plot(d_loc[0], d_loc[1], 'go', markeredgecolor='white',
         markeredgewidth=0.5)

# Label the points
ax1.text(a_loc[0] + x_eps, a_loc[1] - y_eps, '2',
         horizontalalignment='right', verticalalignment='top')
ax1.text(b_loc[0] + x_eps, b_loc[1] - y_eps, '4',
         horizontalalignment='right', verticalalignment='top')
ax1.text(c_loc[0] + x_eps, c_loc[1] + y_eps, '3',
         horizontalalignment='right', verticalalignment='bottom')
ax1.text(d_loc[0] + x_eps, d_loc[1] + y_eps, '1',
         horizontalalignment='right', verticalalignment='bottom')

# Label the Subplots
ax2.text(0.9, 0.9, '1', horizontalalignment='right',
         verticalalignment='top', transform=ax2.transAxes)
ax3.text(0.9, 0.9, '3', horizontalalignment='right',
         verticalalignment='top', transform=ax3.transAxes)
ax4.text(0.9, 0.9, '2', horizontalalignment='right',
         verticalalignment='top', transform=ax4.transAxes)
ax5.text(0.9, 0.9, '4', horizontalalignment='right',
         verticalalignment='top', transform=ax5.transAxes)


ax_all.set_ylabel('Detrended Bioluminescence')
ax_all.set_xlabel('Time (hrs)')
ax5.set_xlim([0, 68])
ax2.set_ylim([-20, 20])

ax1.set_rasterization_zorder(1)

# fig.tight_layout(**layout_pad)
fig.savefig('r2_vs_decay.svg', dpi=500)

# ax1.set_title('ax1')
# ax2.set_title('ax2')
# ax3.set_title('ax3')
# ax4.set_title('ax4')
# ax5.set_title('ax5')
