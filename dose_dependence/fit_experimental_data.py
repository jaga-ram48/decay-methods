# This file processes and fits decaying sinusoids to the raw
# bioluminescence data of the experimental application of small
# molecules KL001 and Longdaysin in increasing concentration. Data is
# imported from the data directory. Creates several figures to display
# the data in various formats. Saves the resulting fits in the
# experimental_fits directory.

import pandas as pd
import numpy as np
from scipy import stats

genes = ['bmal1', 'per2']
drugs = ['KL001', 'longdaysin']

frames = []
frame_list = []

# Loop over each of the four files, creating a dataframe. The
# highest concentrations of each drug lead to increasingly poor rhythms,
# and therefore these concentrations are ignored in parts of the fitting
# and analysis

for drug in drugs:
    for gene in genes:
        # Skipping first two columns (71uM set)
        frames += [pd.DataFrame.from_csv('experimental_data/DataTH_' +
                                         gene + '_' + drug +
                                         '.csv').iloc[:,2:].T]
        frame_list += [(drug, gene)]


sampling_period = np.diff(frames[0].columns.values).mean()


from methods.sinusoid_estimation import fit_data
def fit(row):
    return fit_data(row.values, sampling_period, outliers=False, full=False)


frame_fits = []
# Loop over the frames, appling the fit_ts function and saving the
# results
for frame in frames:
    # Drop the first three datapoints from the start of each trajectory
    fitted = frame.iloc[:,3:].apply(fit, axis=1)
    fitted = fitted.reset_index()
    for i, item in enumerate(fitted['index']):
        # Here we have to correct a strange naming inconsistency,
        # allowing both columns to have the same index.
        if item[-2:] == '.1':
            fitted['index'][i] = float(item[:-4])
        else:  fitted['index'][i] = float(item[:-2])

    frame_fits += [fitted]

# save output fits
for iden, fitted in zip(frame_list, frame_fits): 
    drug = iden[0]
    gene = iden[1]
    fitted.to_csv('experimental_fits/fits_' + drug + '_' + gene +
                  '.csv')


# The 24uM experiment of KL001 in Per2 cells had very fast damping and
# poor sinusoidal fits, and was therefore removed from further analysis
frame_fits[1] = frame_fits[1].iloc[2:]


# First we add the full fit results to the fitted objects
def fit_full(row):
    return fit_data(row.values, sampling_period, outliers=False,
                    full=True)
frame_full_fits = []
for frame in frames:
    fitted = frame.iloc[:,3:].apply(fit_full, axis=1)
    fitted = fitted.reset_index()
    for i, item in enumerate(fitted['index']):
        if item[-2:] == '.1':
            fitted['index'][i] = float(item[:-4])
        else: fitted['index'][i] = float(item[:-2])
    frame_full_fits += [fitted]


# Import some plotting shortcuts and formatting functions
from CommonFiles.PlotOptions import (PlotOptions, layout_pad,
                                     lighten_color, plot_gray_zero,
                                     solarized)
PlotOptions(uselatex=True)
import matplotlib.pylab as plt
from CommonFiles.PlotOptions import solarized

sol_colors = ['yellow', 'orange', 'violet', 'blue']
colors = [solarized[c] for c in sol_colors]
iden_list = [r'KL001, {\itshape Bmal1-dLuc}',
             r'KL001, {\itshape Per2-dLuc}',
             r'Longdaysin, {\itshape Bmal1-dLuc}',
             r'Longdaysin, {\itshape Per2-dLuc}']




# Just a normal plot of the bioluminescence data

# fig, axmatrix = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1)
ax3 = fig.add_subplot(223, sharex=ax1)
ax4 = fig.add_subplot(224, sharex=ax1, sharey=ax3)

plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax2.get_yticklabels(), visible=False)
plt.setp(ax4.get_yticklabels(), visible=False)

axlist = [ax1, ax3, ax2, ax4]

for frame, iden, color, ax in zip(frame_full_fits, iden_list, colors,
                                  axlist):
    frame['index'] = frame['index'].astype(str)
    grouped = frame.set_index(['index', [0,1]*9])
    amounts = np.array(grouped.index.levels[0].tolist())
    inds = np.array(amounts, dtype=float).argsort()
    for amount, lighten in zip(amounts[inds][:-2],
                               np.linspace(0, 0.8, len(amounts))):
        for row in grouped.loc[amount].iterrows():
            row = row[1]
            y = row.x_raw
            x = np.arange(len(y))*sampling_period
            ax.plot(x, y, color=lighten_color(color, lighten))

    ax.text(0.92, 0.92, iden, verticalalignment='top',
            horizontalalignment='right', transform=ax.transAxes)
    # ax.set_ylim([-1, 1])

import matplotlib.lines as mlines
lines = []
labels = []
for amount, lighten in zip(amounts[inds][:-2],
                           np.linspace(0, 0.8, len(amounts))):
    lines += [mlines.Line2D([], [], color=lighten_color('0.47', lighten))]
    labels += [amount + '$\mu$M']


ax1.legend(lines, labels, ncol=2)

ax3.set_ylim([0, 800])

ax.set_xlim([0, len(y)*sampling_period])
ax3.set_xlabel('Time (hrs)')
ax4.set_xlabel('Time (hrs)')
ax1.set_ylabel('Bioluminescence')
ax3.set_ylabel('Bioluminescence')
fig.tight_layout(**layout_pad)



fig = plt.figure(figsize=(2.37, 1.63))
ax = fig.add_subplot(111)

for frame, iden, color in zip(frame_fits, iden_list, colors):
    grouped = frame.groupby('index')
    means = grouped.aggregate(lambda x: np.mean(x, axis=0))
    error = grouped.aggregate(lambda x: stats.sem(x, axis=0))

    out = ax.plot(range(len(means.R2)), means.R2,
                      marker='o', color=color,
                      markeredgecolor='none', linestyle='--',
                      label=iden, zorder=2)
    out = ax.errorbar(range(len(means.R2)), means.R2, yerr=error.R2,
                      ecolor='#262626', elinewidth=0.5, capthick=0.5,
                      zorder=1, linestyle='none')

ax.legend(ncol=1, loc='lower left', numpoints=2, prop={'size':6})
ax.set_xlabel(r'Drug Concentration ($\mu$M)')
ax.set_ylabel(r'$R^2$')
ax.set_xticks(range(len(means.decay)))
ax.set_xlim([-0.5, len(means.decay) - 0.5])
ax.set_xticklabels([str(i) for i in means.index])
fig.tight_layout(**layout_pad)






# Main figure!

import matplotlib

mainfig = plt.figure(figsize=(7.25, 3))
gs_left = matplotlib.gridspec.GridSpec(2,2)
gs_right = matplotlib.gridspec.GridSpec(2,1)

gs_left.update(right=0.5)
gs_right.update(left=0.6)

ax_all = mainfig.add_subplot(gs_left[:,:])
ax_all.spines['top'].set_color('none')
ax_all.spines['bottom'].set_color('none')
ax_all.spines['left'].set_color('none')
ax_all.spines['right'].set_color('none')
ax_all.tick_params(labelcolor='w', top='off', bottom='off', left='off',
                   right='off')

ax1 = mainfig.add_subplot(gs_left[0,0])
ax2 = mainfig.add_subplot(gs_left[0,1], sharex=ax1, sharey=ax1)
ax3 = mainfig.add_subplot(gs_left[1,0], sharex=ax1, sharey=ax1)
ax4 = mainfig.add_subplot(gs_left[1,1], sharex=ax1, sharey=ax1)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax2.get_yticklabels(), visible=False)
plt.setp(ax4.get_yticklabels(), visible=False)
axmatrix_ts = np.array([[ax1, ax2], [ax3, ax4]])



iden_list = [r'KL001, {\itshape Bmal1-dLuc}',
             r'KL001, {\itshape Per2-dLuc}',
             r'Longdaysin, {\itshape Bmal1-dLuc}',
             r'Longdaysin, {\itshape Per2-dLuc}']




# Here we plot the bioluminescence data, but normalize the data using
# the fitted period, phase, and amplitude information. This process
# leaves the decay rate in the data untouched, which highlights the
# differences between the KL001 and Longdaysin trajectories 
#
# Next we constuct the plot
# fig, axmatrix = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)

# Loop over each experimental condition, giving a different color
for frame, iden, color, ax in zip(frame_full_fits, iden_list, colors,
                                  axmatrix_ts.flatten()):
    frame['index'] = frame['index'].astype(str)
    grouped = frame.set_index(['index', [0,1]*9])
    amounts = np.array(grouped.index.levels[0].tolist())
    inds = np.array(amounts, dtype=float).argsort()

    # Loop over the experimental conditions, giving different plotting
    # options
    for amount, lighten in zip(amounts[inds][:-2],
                               np.linspace(0, 0.8, len(amounts))):

        # Loop over each trajectory
        for row in grouped.loc[amount].iterrows():
            row = row[1]
            
            # Get the baseline-subtracted data
            y_adj = row.x_detrend

            # Get the fitted parameters
            amplitude = row.amplitude
            period = row.period
            phase = row.phase

            # Get a time-series for x-values
            x = np.arange(len(y_adj))*sampling_period
            
            if np.all(np.abs(y_adj/amplitude < 1.)):
                x = x/period + phase/(2*np.pi)
                ax.plot(x - x[0], y_adj/amplitude,
                        color=lighten_color(color, lighten))

    # ax.set_title(iden)
    ax.text(0.95, 0.05, iden, verticalalignment='bottom',
            horizontalalignment='right', transform=ax.transAxes,
            size='small')
    ax.set_ylim([-1, 1])
    ax.set_xlim([0, 5])
    plot_gray_zero(ax, zorder=0)

ax_all.set_xlabel('Phase (days)')
ax_all.set_ylabel('Bioluminescence')

import matplotlib.lines as mlines
lines = []
labels = []
for amount, lighten in zip(amounts[inds][:-2],
                           np.linspace(0, 0.8, len(amounts))):
    lines += [mlines.Line2D([], [], color=lighten_color('0.47', lighten))]
    labels += [amount + '$\mu$M']


axmatrix_ts[0,0].legend(lines, labels, ncol=2)



# Here we plot the dose dependent increase in decay rate for each of the
# 4 systems (2 drugs, 2 reporters)
# colors = ['#FF4844', '#FF9C44', '#2C97A1', '#36CC40']

# fig, axmatrix = plt.subplots(nrows=2, sharex=True, figsize=(2.37, 2.37))

ax1 = mainfig.add_subplot(gs_right[0,0])
ax2 = mainfig.add_subplot(gs_right[1,0], sharex=ax1)
plt.setp(ax2.get_xticklabels(), visible=False)
axmatrix_kd = np.array([ax1, ax2])


for frame, iden, color in zip(frame_fits, iden_list, colors):
    grouped = frame.groupby('index')
    means = grouped.aggregate(lambda x: np.mean(x, axis=0))
    error = grouped.aggregate(lambda x: stats.sem(x, axis=0))

    out = ax1.plot(range(len(means.decay)), means.decay,
                      marker='o', color=color,
                      markeredgecolor='none', linestyle='--',
                      label=iden, zorder=2)
    out = ax1.errorbar(range(len(means.decay)), means.decay,
                       yerr=error.decay, ecolor='#262626',
                       elinewidth=0.5, capthick=0.5, zorder=1,
                       linestyle='none')

# ax1.legend(ncol=1, loc='upper left', numpoints=2, prop={'size':6})
# ax1.set_xlabel(r'Drug Concentration ($\mu$M)')
ax1.set_ylabel(r'Damping Rate $\left(\nicefrac{1}{\mathrm{hrs}}\right)$')

# ax1.set_xticks(range(len(means.D)))
# ax1.set_xlim([-0.5, len(means.D) - 0.5])
# ax1.set_xticklabels([str(i) for i in means.index])

# Same plot, but this time dose dependent increase in Period

for frame, iden, color in zip(frame_fits, iden_list, colors):
    grouped = frame.groupby('index')
    means = grouped.aggregate(lambda x: np.mean(x, axis=0))
    error = grouped.aggregate(lambda x: stats.sem(x, axis=0))

    out = ax2.plot(range(len(means.period)), means.period,
                      marker='o', color=color,
                      markeredgecolor='none', linestyle='--',
                      label=iden, zorder=2)
    out = ax2.errorbar(range(len(means.period)), means.period,
                       yerr=error.period, ecolor='#262626',
                       elinewidth=0.5, capthick=0.5, zorder=1,
                       linestyle='none')

ax2.legend(ncol=1, loc='upper left', numpoints=2, prop={'size':6})
ax2.set_xlabel(r'Drug Concentration ($\mu$M)')
ax2.set_ylabel(r'Period (hrs)')

ax2.set_xticks(range(len(means.decay)))
ax2.set_xlim([-0.5, len(means.decay) - 0.5])
ax2.set_xticklabels([str(i) for i in means.index])

# Interestingly, the natural decay rate for Per2-dLuc is lower than
# Bmal1-dLuc, which likely reflects an underlying difference in the
# clock dynamics of each reporter system.

# mainfig.tight_layout(**layout_pad)
gs_left.update(wspace=.075, left=0.075)
gs_right.update(right=.99)







plt.show()
