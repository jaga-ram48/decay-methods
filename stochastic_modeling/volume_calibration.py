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



fig = plt.figure(figsize=(2.5, 1.875))
ax_vol = fig.add_subplot(111)

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
ax_vol.set_ylabel(r'Damping Rate $\left(\nicefrac{1}{\mathrm{hrs}}\right)$')

fig.tight_layout(**layout_pad)

plt.show()
