# Function to plot a single perturbation strength to both vdcn and vac1p
# parameters, showing an example of the decay and period rate changes
# from each change.

import numpy as np
from scipy import stats
import cPickle as pickle

from CommonFiles.pBase import pBase
from CommonFiles.Models.degModelFinal import create_class

from CommonFiles.Models.DegModelStoch import simulate_stoch

base_control = create_class()

# Here we allow the stochastic simulations to be either run or loaded
# from a previous run. Here we simply load the data from a previous run
# to cut down on computational cost.
# opt = 'run'
opt = 'load'

if opt == 'load':
    with open('data/single_simulation_results.p', 'rb') as f:
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

else: 
    # Similar to the cluster script, we have to first calculate the limit
    # cycle solutions to the models at a variety of different knockdown
    # strengths

    def param_reduction(param, amount, base_list):
        t_param = np.array(base_control.paramset)
        t_param[base_control.pdict[param]] *= (1-amount)
        tbase = pBase(base_control.model, t_param, base_list[-1].y0)
        try:
            tbase.approxY0(tout=1000, tol=1E-5)
            tbase.solveBVP()
            assert tbase.findstationary() == 0
            base_list += [tbase]
        except Exception: pass
            # base_list += [np.nan]

    vac1p_base_list = [base_control]
    vac1p_vals = np.linspace(0, 0.95, num=100)
    for i in vac1p_vals[1:]:
        param_reduction('vaC1P', i, vac1p_base_list)

    vdcn_base_list = [base_control]
    vdcn_vals = np.linspace(0, 0.95, num=100)
    for i in vdcn_vals[1:]:
        param_reduction('vdCn', i, vdcn_base_list)

    # Moderate perturbations for both parameters
    base_vac1p = vac1p_base_list[65]
    base_vdcn = vdcn_base_list[65]

    # These parameters are taken from the cluster runs and volume
    # calibration

    vol = 225
    periods = 10
    ntraj = 1000

    # Estimate decay parameter from stochastic trajectories
    ts_c, traj_control = simulate_stoch(base_control, vol,
                                        t=periods*base_control.y0[-1],
                                        traj=ntraj,
                                        increment=base_control.y0[-1]/100)

    ts_p, traj_vac1p = simulate_stoch(base_vac1p, vol,
                                      t=periods*base_control.y0[-1],
                                      traj=ntraj,
                                      increment=base_control.y0[-1]/100)

    ts_n, traj_vdcn = simulate_stoch(base_vdcn, vol,
                                     t=periods*base_control.y0[-1],
                                     traj=ntraj,
                                     increment=base_control.y0[-1]/100)

    # traj_control = traj_control.mean(0)
    # traj_vac1p = traj_vac1p.mean(0)
    # traj_vdcn = traj_vdcn.mean(0)
    
    traj_dict = {
        'ts'       : ts_c,
        'control'  : traj_control,
        'vac1p'    : traj_vac1p,
        'vdcn'     : traj_vdcn,
        'vdcn_y0'  : base_vdcn.y0,
        'vdcn_p'   : base_vdcn.paramset,
        'vac1p_y0' : base_vac1p.y0,
        'vac1p_p'  : base_vac1p.paramset,
    }

    with open('data/single_simulation_results.p', 'wb') as f:
        pickle.dump(traj_dict, f)




# Here we estimate the decay parameters associated with the stochastic
# simulation by fitting a continuous model approximation (see previous
# paper)
from CommonFiles.StochDecayEstimator import StochDecayEstimator
trans = 0

Estimator_control = StochDecayEstimator(ts_c[trans:],
                                        traj_control[:,trans:,:].mean(0),
                                        base_control)
Estimator_vac1p = StochDecayEstimator(ts_c[trans:],
                                      traj_vac1p[:,trans:,:].mean(0),
                                      base_vac1p)
Estimator_vdcn = StochDecayEstimator(ts_c[trans:],
                                     traj_vdcn[:,trans:,:].mean(0),
                                     base_vdcn)



import pandas as pd
# We next loop over the fitted data, saved as a pickled pandas dataframe
genes = ['bmal1', 'per2']
drugs = ['KL001', 'longdaysin']
frame_list = []
frame_fits = []
for drug in drugs:
    for gene in genes:
        frame_list += [(drug, gene)]
        frame_fits += [pd.read_csv('../dose_dependence/experimental_fits/fits_' + drug +
                                   '_' + gene + '.csv')]



iden_list = [r'KL001, {\itshape Bmal1-dLuc}',
             r'KL001, {\itshape Per2-dLuc}',
             r'Longdaysin, {\itshape Bmal1-dLuc}',
             r'Longdaysin, {\itshape Per2-dLuc}']

d_max = 0.04
p_max = 40


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
vac1p_means_trunc = vac1p_means[in_plot]
vac1p_error_trunc = vac1p_error[in_plot]

# The stochastic simulation of the vdcn parameter diverges from the
# continuous approximation for high knockdown amounts (the model
# steadily approaches a steady state) so here we restrict the plots to
# fits that have a high R2.
high_r2 = vdcn_means.R > 0.9
vdcn_means_trunc = vdcn_means[high_r2]
vdcn_error_trunc = vdcn_error[high_r2]


# 
# # Print relevant estimated quantities to the screen
# 
# print "Periods (Deterministic)"
# print "-------"
# print  "Control:\t{0:4.3f}".format(base_control.y0[-1])
# print "vaC1P KD:\t{0:5.3f}".format(base_vac1p.y0[-1])
# print  "vdCN KD:\t{0:5.3f}".format(base_vdcn.y0[-1])
# 
# print "Periods (Stochastic)"
# print "-------"
# print "Control:\t{0:4.3f}".format(
#     Estimator_control.sinusoid_params['period'])
# print "vaC1P KD:\t{0:5.3f}".format(
#     Estimator_vac1p.sinusoid_params['period'])
# print  "vdCN KD:\t{0:5.3f}".format(
#     Estimator_vdcn.sinusoid_params['period'])
# 
# print "Phase Diffusivities"
# print "-------------------"
# print  "Control:\t{0:4.3f}".format(Estimator_control.decay)
# print "vaC1P KD:\t{0:5.3f}".format(Estimator_vac1p.decay)
# print  "vdCN KD:\t{0:5.3f}".format(Estimator_vdcn.decay)
# 
# print "R2 Values"
# print "-------------------"
# print  "Control:\t{0:4.3f}".format(Estimator_control.r2)
# print "vaC1P KD:\t{0:5.3f}".format(Estimator_vac1p.r2)
# print  "vdCN KD:\t{0:5.3f}".format(Estimator_vdcn.r2)
# 
# Plot deterministic and stochastic trajectories
import matplotlib.pyplot as plt
import matplotlib
from CommonFiles.PlotOptions import PlotOptions, layout_pad, color_range
PlotOptions(uselatex=True)

from CommonFiles.PlotOptions import solarized
color_names = ['yellow', 'orange', 'red', # Hot
               'violet', 'blue', 'cyan'] # Cool
colors = [solarized[c] for c in color_names]
# 
# colors = list(color_range(base_control.NEQ))
# 
# fig = plt.figure()
# ax = fig.add_subplot(111)
# for i in xrange(base_control.NEQ):
#     ax.set_title('Control')
#     ax.plot(ts_c, traj_control[:,:,i].mean(0), ':', color=colors[i])
#     ax.plot(Estimator_control.x, Estimator_control.x_bar[:,i], '--',
#             color=colors[i])
# 
# fig = plt.figure()
# ax = fig.add_subplot(111)
# for i in xrange(base_control.NEQ):
#     kd_vac1p = (base_vac1p.parambykey('vaC1P') /
#                 base_control.parambykey('vaC1P'))
#     ax.set_title('vaC1P {0:0.3f}'.format(1-kd_vac1p))
#     ax.plot(ts_c, traj_vac1p[:,:,i].mean(0), ':', color=colors[i])
#     ax.plot(Estimator_vac1p.x, Estimator_vac1p.x_bar[:,i], '--',
#             color=colors[i])
# 
# fig = plt.figure()
# ax = fig.add_subplot(111)
# for i in xrange(base_control.NEQ):
#     kd_vdcn = (base_vdcn.parambykey('vdCn') /
#                base_control.parambykey('vdCn'))
#     ax.set_title('vdCn {0:0.3f}'.format(1-kd_vdcn))
#     ax.plot(ts_c, traj_vdcn[:,:,i].mean(0), ':', color=colors[i])
#     ax.plot(Estimator_vdcn.x,
#             Estimator_vdcn.x_bar[:,i], '--',
#             color=colors[i])
# 
# fig.tight_layout(**layout_pad)
# 

mainfig = plt.figure(figsize=(7.25, 3))

gs_left = matplotlib.gridspec.GridSpec(2,3)
gs_right = matplotlib.gridspec.GridSpec(1,1)

gs_left.update(right=0.6, left=0.05, top=0.925, wspace=0.05)
gs_right.update(left=0.675, right=0.99, top=0.925)

ax1 = mainfig.add_subplot(gs_left[0,0])
ax2 = mainfig.add_subplot(gs_left[0,1], sharex=ax1, sharey=ax1)
ax3 = mainfig.add_subplot(gs_left[0,2], sharex=ax1, sharey=ax1)
ax4 = mainfig.add_subplot(gs_left[1,0], sharex=ax1, sharey=ax1)
ax5 = mainfig.add_subplot(gs_left[1,1], sharex=ax1, sharey=ax1)
ax6 = mainfig.add_subplot(gs_left[1,2], sharex=ax1, sharey=ax1)
axmatrix = np.array([[ax1, ax2, ax3], [ax4, ax5, ax6]])
for ax in axmatrix[0,:]:
    plt.setp(ax.get_xticklabels(), visible=False)
for ax in axmatrix[:,1:].flat:
    plt.setp(ax.get_yticklabels(), visible=False)


ax_kd = mainfig.add_subplot(gs_right[0,0])

# fig, axmatrix = plt.subplots(nrows=2, ncols=3, sharey=True, sharex=True,
#                              figsize=(3.425, 2.75))



# colors = ['#FF4844', '#2C97A1']

# Control
amp = Estimator_control._cos_dict['amp'][0]
dec = -Estimator_control.decay
bas = Estimator_control._cos_dict['baseline'][0]
per = Estimator_control.sinusoid_params['period']
axmatrix[1,0].plot(ts_c, traj_control[:,:,0].mean(0), color='k')
axmatrix[1,0].plot(ts_c,  amp*np.exp(dec*ts_c) + bas, '--', color='k')
axmatrix[1,0].plot(ts_c, -amp*np.exp(dec*ts_c) + bas, '--', color='k')

disp = '\\begin{{align*}}d &= {0:0.3f}\\\\[-1ex]T &= {1:0.1f}\\end{{align*}}'.format(-dec, per)
axmatrix[1,0].text(0.92, 0.9, disp, horizontalalignment='right',
                 verticalalignment='top',
                 transform=axmatrix[1,0].transAxes, fontsize=7)

# vdcn
amp = Estimator_vdcn._cos_dict['amp'][0]
dec = -Estimator_vdcn.decay
bas = Estimator_vdcn._cos_dict['baseline'][0]
per = Estimator_vdcn.sinusoid_params['period']
axmatrix[1,1].plot(ts_c, traj_vdcn[:,:,0].mean(0), color=colors[2])
axmatrix[1,1].plot(ts_c,  amp*np.exp(dec*ts_c) + bas, '--',
                 color=colors[2])
axmatrix[1,1].plot(ts_c, -amp*np.exp(dec*ts_c) + bas, '--',
                 color=colors[2])

disp = '\\begin{{align*}}d &= {0:0.3f}\\\\[-1ex]T &= {1:0.1f}\\end{{align*}}'.format(-dec, per)
axmatrix[1,1].text(0.92, 0.9, disp, horizontalalignment='right',
                 verticalalignment='top',
                 transform=axmatrix[1,1].transAxes, fontsize=7)
          #
# vac1p
amp = Estimator_vac1p._cos_dict['amp'][0]
dec = -Estimator_vac1p.decay
bas = Estimator_vac1p._cos_dict['baseline'][0]
per = Estimator_vac1p.sinusoid_params['period']
axmatrix[1,2].plot(ts_c, traj_vac1p[:,:,0].mean(0), color=colors[5])
axmatrix[1,2].plot(ts_c,  amp*np.exp(dec*ts_c) + bas, '--',
                 color=colors[5])
axmatrix[1,2].plot(ts_c, -amp*np.exp(dec*ts_c) + bas, '--',
                 color=colors[5])

disp = '\\begin{{align*}}d &= {0:0.3f}\\\\[-1ex]T &= {1:0.1f}\\end{{align*}}'.format(-dec, per)
axmatrix[1,2].text(0.92, 0.9, disp, horizontalalignment='right',
                 verticalalignment='top',
                 transform=axmatrix[1,2].transAxes, fontsize=7)


axmatrix[1,0].set_ylim([0, 1.75])
axmatrix[1,0].set_xticks([0, 100, 200])

axmatrix[1,1].set_xlabel('Time (hrs)')
# fig.tight_layout(**layout_pad)



axmatrix[0,0].plot(ts_c, traj_control[0,:,0], 'k')
axmatrix[0,1].plot(ts_c, traj_vdcn[0,:,0], color=colors[2])
axmatrix[0,2].plot(ts_c, traj_vac1p[0,:,0], color=colors[5])



axmatrix[0,0].set_ylabel('Single Cell Expression')
axmatrix[1,0].set_ylabel('Population Expression')

axmatrix[0,0].set_title('Control', size='medium')
axmatrix[0,1].set_title('KL001', size='medium')
axmatrix[0,2].set_title('Longdaysin', size='medium')




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


for frame, iden, color in zip(frame_fits, iden_list, colors[0:2] +
                              colors[3:5]):

    frame = frame[frame.R2 > 0.85]
    grouped = frame.groupby('index')
    means = grouped.aggregate(lambda x: np.mean(x, axis=0))
    error = grouped.aggregate(lambda x: stats.sem(x, axis=0))

    in_plot = (means.decay < d_max) & (means.period < p_max)
    means_red = means[in_plot]
    error_red = error[in_plot]
    # means_red = means[in_plot]/means.iloc[0]
    # error_red = error[in_plot]/means.iloc[0]

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
leg = ax_kd.legend(handles, labels, ncol=2, loc='upper center', numpoints=1, prop={'size':6})

# Relevant window
# ax_kd.set_xlim(20, 8903344)
ax_kd.set_ylim(0, 0.04)

ax_kd.set_ylabel(r'Damping Rate $\left(\nicefrac{1}{\mathrm{hrs}}\right)$')
ax_kd.set_xlabel('Period (hrs)')



plt.show()

