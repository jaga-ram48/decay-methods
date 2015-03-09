import pandas as pd
import numpy as np
from scipy import stats
import matplotlib

from CommonFiles.PlotOptions import PlotOptions, layout_pad, blue, green
PlotOptions(uselatex=True)
import matplotlib.pylab as plt

# Load the pickled dataframe
decay_f = pd.read_csv('stochastic_model_data/volume_results.csv')
vols = decay_f.V

# Here we group the data into similar groups, since for each volume 10
# independent groups of trajectories were calculated. This allows us to
# estimate the true error in estimated decay, amplitude, period, etc.
decay_group = decay_f.drop(['V'], 1).T.groupby(lambda x: x[0])

# Calcalate mean and SEM of the volume-decay data. 
means_all = decay_group.aggregate(lambda x,b: np.mean(x, axis=0), 1).T
error_all = decay_group.aggregate(lambda x,b: stats.sem(x, axis=0), 1).T

# Only want to consider good fits, where the R2 is larger than 0.9
high_r2 = means_all.R > 0.9
vols_reg = vols[high_r2]
means_all_reg = means_all[high_r2]
error_all_reg = error_all[high_r2]

# Here we're only looking at decay rate
v_means = means_all_reg.D
v_error = error_all_reg.D

# fit a linear regression model
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
log_x = np.log10(vols_reg).values
log_means = np.log10(v_means).values
log_error = abs(np.log10(v_error).values)
x = sm.add_constant(log_x)
wls_model = sm.WLS(log_means, x, weights=log_error)
wls_fit = wls_model.fit()

# Confidence region
prstd, iv_l, iv_u = wls_prediction_std(wls_fit)

regression = 10**(wls_fit.fittedvalues)
regression_lb = 10**(iv_l)
regression_ub = 10**(iv_u)

# Decay ratio of experiments, taken from the process_data.py script.
exp_decay = 0.01506

# Interpolate for mean and error of the estimated volume. (Interpolate
# in log domain for y -> x)
log_exp_decay = np.log10(exp_decay)
log_fitted_decay = wls_fit.fittedvalues

sort_inds = log_fitted_decay.argsort()


# Calculate the correct volume calibration (as well as expected upper
# and lower bounds) from the linear model
log_vol = np.interp(log_exp_decay, log_fitted_decay[sort_inds],
                    log_x[sort_inds])
log_vol_lb = np.interp(log_exp_decay, iv_l[sort_inds], log_x[sort_inds])
log_vol_ub = np.interp(log_exp_decay, iv_u[sort_inds], log_x[sort_inds])

import cPickle as pickle

from CommonFiles.pBase import pBase
from CommonFiles.Models.degModelFinal import create_class

base_control = create_class()


with open('stochastic_model_data/single_simulation_results.p', 'rb') as f:
    traj_dict = pickle.load(f)

ts_c         = traj_dict['ts']
traj_control = traj_dict['control']
traj_vac1p   = traj_dict['vac1p']
traj_vdcn    = traj_dict['vdcn']
vdcn_y0      = traj_dict['vdcn_y0']
vdcn_p       = traj_dict['vdcn_p']
vac1p_y0     = traj_dict['vac1p_y0']
vac1p_p      = traj_dict['vac1p_p']

base_vdcn = pBase(base_control.model, vdcn_p, vdcn_y0)
base_vac1p = pBase(base_control.model, vac1p_p, vac1p_y0)




# Here we estimate the decay parameters associated with the stochastic
# simulation by fitting a continuous model approximation (see previous
# paper)
from CommonFiles.StochDecayEstimator import StochDecayEstimator
trans = 0

Estimator_control = StochDecayEstimator(ts_c[trans:],
                                        traj_control[trans:,:],
                                        base_control)
Estimator_vac1p = StochDecayEstimator(ts_c[trans:],
                                      traj_vac1p[trans:,:], base_vac1p)
Estimator_vdcn = StochDecayEstimator(ts_c[trans:],
                                     traj_vdcn[trans:,:], base_vdcn)













# Plot the resulting calibration curve
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(1, 2, width_ratios=[1.5,1])
gs0 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0],
                                       height_ratios=[1,1.5],
                                       hspace=0.25)
gs1 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1],
                                       hspace=0.15)


fig = plt.figure(figsize=(3.425, 3.425))
ax_vol = fig.add_subplot(gs0[0])
ax_kd  = fig.add_subplot(gs0[1])
ax_c   = fig.add_subplot(gs1[0])
ax_vd  = fig.add_subplot(gs1[1], sharex=ax_c, sharey=ax_c)
ax_va  = fig.add_subplot(gs1[2], sharex=ax_c, sharey=ax_c)

plt.setp(ax_c.get_xticklabels(), visible=False)
plt.setp(ax_vd.get_xticklabels(), visible=False)
# plt.setp(ax_c.get_yticklabels(), visible=False)
# plt.setp(ax_vd.get_yticklabels(), visible=False)
# plt.setp(ax_va.get_yticklabels(), visible=False)


from matplotlib.ticker import FormatStrFormatter
ax_vol.plot(vols_reg, regression, '-', color='#C8C8C8', zorder=0)
ax_vol.plot(vols_reg, regression_lb, ':', color='#C8C8C8')
ax_vol.plot(vols_reg, regression_ub, ':', color='#C8C8C8')

out = ax_vol.plot(vols_reg, v_means, marker='o', color=blue,
              markeredgecolor='none', linestyle='none', zorder=2)
out = ax_vol.errorbar(vols_reg, v_means, yerr=v_error, ecolor='#262626',
                  elinewidth=0.5, capthick=0.5, zorder=1,
                  linestyle='none')

low_r2 = means_all.R < 0.9
out = ax_vol.plot(vols[low_r2], means_all[low_r2].D, marker='o',
              color='gray', markeredgecolor='none', linestyle='none',
              zorder=2)
out = ax_vol.errorbar(vols[low_r2], means_all[low_r2].D,
                  yerr=error_all[low_r2].D, ecolor='gray',
                  elinewidth=0.5, capthick=0.5, zorder=1,
                  linestyle='none')

ax_vol.axhline(exp_decay, linestyle='--', color=green, zorder=0)
ax_vol.text(98, exp_decay - 0.001, '$d = {0:0.3f}$'.format(exp_decay),
            horizontalalignment='left', verticalalignment='top', fontsize=7)

ax_vol.set_xscale('log')
ax_vol.set_yscale('log')
ax_vol.get_yaxis().set_major_formatter(FormatStrFormatter("%.2f"))
ax_vol.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax_vol.set_ylim([0.006, 0.035])
ax_vol.set_xlim([90, 550])
ax_vol.set_xticks([100, 200, 300, 400, 500])
ax_vol.set_yticks([0.01, 0.02, 0.03])
ax_vol.set_xlabel(r'$\Omega$')
ax_vol.set_ylabel(r'Decay $\left(\nicefrac{1}{\mathrm{hrs}}\right)$')


# Next is the parameter kd plot

# Load data from previous runs
vac1p = pd.read_csv('stochastic_model_data/vac1p_results.csv')
vdcn = pd.read_csv('stochastic_model_data/vdcn_results.csv')

# Group the data by replication number, find the means and SEM
vac1p_group = vac1p.T.groupby(lambda x: x[0])
vac1p_means = vac1p_group.mean().T
vac1p_error = vac1p_group.aggregate(lambda x: stats.sem(x, axis=0)).T

vdcn_group = vdcn.T.groupby(lambda x: x[0])
vdcn_means = vdcn_group.mean().T
vdcn_error = vdcn_group.aggregate(lambda x: stats.sem(x, axis=0)).T

# Here we trim the data to only show relevant results

# The model's period after changing vac1p (longdaysin) extends for a
# very long range towards the bifurcation, here we only show the
# relevant section from 24 -> 38 hrs.
in_plot = vac1p_means['T'] < 38.
vac1p_means_trunc = vac1p_means[in_plot]/vac1p_means.iloc[0]
vac1p_error_trunc = vac1p_error[in_plot]/vac1p_means.iloc[0]

# The stochastic simulation of the vdcn parameter diverges from the
# continuous approximation for high knockdown amounts (the model
# steadily approaches a steady state) so here we restrict the plots to
# fits that have a high R2.
high_r2 = vdcn_means.R > 0.9
vdcn_means_trunc = vdcn_means[high_r2]/vdcn_means.iloc[0]
vdcn_error_trunc = vdcn_error[high_r2]/vdcn_means.iloc[0]

from CommonFiles.PlotOptions import solarized
color_names = ['yellow', 'orange', 'red', # Hot
               'violet', 'blue', 'cyan'] # Cool
colors = [solarized[c] for c in color_names]


# Create the plot
# Here we plot the simulation results
out = ax_kd.plot(vac1p_means_trunc['B'], vac1p_means_trunc.D, marker='o',
              markeredgecolor='none', linestyle=':', zorder=2,
              color=colors[5],
              label=r'Longdaysin, {\itshape in silico}')
out = ax_kd.errorbar(vac1p_means_trunc['B'], vac1p_means_trunc.D,
                  yerr=vac1p_error_trunc.D, ecolor='#DBDBDB',
                  elinewidth=0.5, capthick=0.5, zorder=1,
                  linestyle='none')

out = ax_kd.plot(vdcn_means_trunc['T'], vdcn_means_trunc.D, marker='o',
              color=colors[2], markeredgecolor='none', linestyle=':',
              zorder=2, label=r'KL001, {\itshape in silico}')
out = ax_kd.errorbar(vdcn_means_trunc['T'], vdcn_means_trunc.D,
                  yerr=vdcn_error_trunc.D, ecolor='#DBDBDB',
                  elinewidth=0.5, capthick=0.5, zorder=1,
                  linestyle='none')


# We next loop over the fitted data, saved as a pickled pandas dataframe
genes = ['bmal1', 'per2']
drugs = ['KL001', 'longdaysin']
frame_list = []
frame_fits = []
for drug in drugs:
    for gene in genes:
        frame_list += [(drug, gene)]
        frame_fits += [pd.read_csv('experimental_fits/fits_' + drug +
                                   '_' + gene + '.csv')]



iden_list = [r'KL001, {\itshape Bmal1-dLuc}',
             r'KL001, {\itshape Per2-dLuc}',
             r'Longdaysin, {\itshape Bmal1-dLuc}',
             r'Longdaysin, {\itshape Per2-dLuc}']

d_max = 0.04
p_max = 40

for frame, iden, color in zip(frame_fits, iden_list, colors[0:2] +
                              colors[3:5]):
    # if iden is iden_list[1]:
    #     frame = frame[4:]
    grouped = frame.groupby('index')
    means = grouped.aggregate(lambda x: np.mean(x, axis=0))
    error = grouped.aggregate(lambda x: stats.sem(x, axis=0))

    in_plot = (means.decay < d_max) & (means.period < p_max)
    means_red = means[in_plot]/means.iloc[0]
    error_red = error[in_plot]/means.iloc[0]

    marker = 's' if 'Bmal1' in iden else 'D'

    out = ax_kd.plot(means_red.period, means_red.decay, marker=marker,
                     markeredgecolor='none', linestyle=':', zorder=2,
                     color=color, label=iden)
    out = ax_kd.errorbar(means_red.period, means_red.decay,
                         xerr=error_red.period, yerr=error_red.decay,
                         ecolor='#DBDBDB', elinewidth=0.5,
                         capthick=0.5, zorder=1, linestyle='none')

handles, labels = ax_kd.get_legend_handles_labels()
labels = [labels[0]] + labels[-2:] + labels[1:4]
handles = [handles[0]] + handles[-2:] + handles[1:4]
leg = ax_kd.legend(handles, labels, ncol=1, loc='upper right', numpoints=1,
                   frameon=True, fancybox=True, prop={'size':6})

# Relevant window
ax_kd.set_xlim(0.95420048035415983, 1.5716183088903344)
ax_kd.set_ylim(0.19269527250455487, 2.5)

ax_kd.set_ylabel('Decay (fold change)')
ax_kd.set_xlabel('Period (fold change)')

# Function to plot a single perturbation strength to both vdcn and vac1p
# parameters, showing an example of the decay and period rate changes
# from each change.

# colors = ['#FF4844', '#2C97A1']

# Control
amp = Estimator_control._cos_dict['amp'][0]
dec = -Estimator_control.decay
bas = Estimator_control._cos_dict['baseline'][0]
per = Estimator_control.sinusoid_params['period']
ax_c.plot(ts_c, traj_control[:,0], color='k')
ax_c.plot(ts_c,  amp*np.exp(dec*ts_c) + bas, '--', color='k')
ax_c.plot(ts_c, -amp*np.exp(dec*ts_c) + bas, '--', color='k')

disp = '\\begin{{align*}}d &= {0:0.3f}\\\\[-1ex]T &= {1:0.1f}\\end{{align*}}'.format(-dec, per)
ax_c.text(0.92, 0.9, disp, horizontalalignment='right',
                 verticalalignment='top',
                 transform=ax_c.transAxes, fontsize=7)

ax_c.text(0.9, 0.1, r'Control, {\itshape in silico}', fontsize=7)

# vac1p
amp = Estimator_vac1p._cos_dict['amp'][0]
dec = -Estimator_vac1p.decay
bas = Estimator_vac1p._cos_dict['baseline'][0]
per = Estimator_vac1p.sinusoid_params['period']
ax_va.plot(ts_c, traj_vac1p[:,0], color=colors[5])
ax_va.plot(ts_c,  amp*np.exp(dec*ts_c) + bas, '--',
                 color=colors[5])
ax_va.plot(ts_c, -amp*np.exp(dec*ts_c) + bas, '--',
                 color=colors[5])

disp = '\\begin{{align*}}d &= {0:0.3f}\\\\[-1ex]T &= {1:0.1f}\\end{{align*}}'.format(-dec, per)
ax_va.text(0.92, 0.9, disp, horizontalalignment='right',
                 verticalalignment='top',
                 transform=ax_va.transAxes, fontsize=7)

ax_va.text(0.9, 0.1, r'Longdaysin, {\itshape in silico}', fontsize=7)

# vdcn
amp = Estimator_vdcn._cos_dict['amp'][0]
dec = -Estimator_vdcn.decay
bas = Estimator_vdcn._cos_dict['baseline'][0]
per = Estimator_vdcn.sinusoid_params['period']
ax_vd.plot(ts_c, traj_vdcn[:,0], color=colors[2])
ax_vd.plot(ts_c,  amp*np.exp(dec*ts_c) + bas, '--',
                 color=colors[2])
ax_vd.plot(ts_c, -amp*np.exp(dec*ts_c) + bas, '--',
                 color=colors[2])

disp = '\\begin{{align*}}d &= {0:0.3f}\\\\[-1ex]T &= {1:0.1f}\\end{{align*}}'.format(-dec, per)
ax_vd.text(0.92, 0.9, disp, horizontalalignment='right',
                 verticalalignment='top',
                 transform=ax_vd.transAxes, fontsize=7)

ax_vd.text(0.9, 0.1, r'KL001, {\itshape in silico}', fontsize=7)

ax_c.set_ylim([0, 1.5])
ax_c.set_xticks([0, 100, 200])
ax_va.set_xlabel('Time (hrs)')

fig.tight_layout(**layout_pad)

plt.show()
