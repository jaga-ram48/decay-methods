import numpy as np
import pandas as pd

from methods.HTExperiment import Experiment


zhang = Experiment('zhang').split()


library = zhang.scaled_blfit_p.join(zhang.names_p)
well_breakdown = library.well.str.extract(
    '(?P<well_row>[A-Za-z]{1})(?P<well_col>\d{1,2})')
library = library.drop('well', 1).join(well_breakdown)
library = library[library.Gene_ID.str.contains(r'0\d{1,6}$')]

def prune_outliers(df, max_dist=6, keys=None):
    """ Function to return a dataframe in which none of the elements of
    'keys' fall between -max_dist < df.key < max_dist """

    if keys == None: keys = ['decay', 'amplitude', 'period']

    return df[((df.loc[:,keys] <  max_dist) & 
               (df.loc[:,keys] > -max_dist)).all(1)]

library = prune_outliers(library, max_dist=8).loc[
    :, ['amplitude', 'decay', 'Gene_ID']]
control = prune_outliers(zhang.scaled_blfit_c, max_dist=8)
# means = library.groupby('Gene_ID').mean()

from sklearn.covariance import EllipticEnvelope, MinCovDet
from scipy.stats import chi2

x_train = control.loc[:, ['amplitude', 'decay']]

S = MinCovDet().fit(x_train)
control_mean = S.location_
inv_cov = np.linalg.pinv(S.covariance_)

# control_mean = x_train.mean()
# inv_cov = np.linalg.pinv(x_train.cov())


def hotelling_tsquared(x):
    """ Function to test the perturbed population x (all with the same
    Gene ID) against the control population in x_test, assuming equal
    covariances """

    # If its a grouped perturbed df, drop the gene_id
    try: x.drop(['Gene_ID'], 1, inplace=True)
    except ValueError: pass

    pert_mean = x.mean()
    mean_diff = control_mean - pert_mean
    T2 = mean_diff.dot(len(x)*inv_cov).dot(mean_diff)
    return 1 - chi2.cdf(T2, 2)

pvals = library.groupby('Gene_ID').apply(hotelling_tsquared)
pert_means = library.groupby('Gene_ID').mean()

# Lets bootstrap a confidence threshold.
# starting with 4 valid copies

import random
random.seed(0)
nboot = 1000

# we want an alpha such that there is a FPR of 5% in the control case.
# (I.e., 95% of control 'gene's would be identified as control)

def get_alpha(n, nboot=1000):
    pvals = np.array([hotelling_tsquared(
        x_train.loc[random.sample(x_train.index, n)])
        for i in xrange(nboot)])
    return np.percentile(pvals, 5)


uniques = library.groupby('Gene_ID').count().amplitude.unique()
uniques.sort()
alpha_dict = {}

for unique in uniques: alpha_dict[unique] = get_alpha(unique, 1000)


from methods.PlotOptions import PlotOptions, layout_pad
PlotOptions(uselatex=True)
import matplotlib.pylab as plt
import matplotlib

# # Older stuff on SVM classifiers
# from sklearn import svm

# x_range = (-6, 6)
# y_range = (-6, 6)
# 
# xx, yy = np.meshgrid(np.linspace(x_range[0], x_range[1], 500), 
#                      np.linspace(y_range[0], y_range[1], 500))

# # fit the model
# clf = EllipticEnvelope(contamination=0.1)
# # clf = svm.OneClassSVM(nu=0.10, kernel="rbf", gamma=0.2)
# 
# clf.fit(x_train.values)
# # y_pred_train = clf.predict(x_train)
# # y_pred_test = clf.predict(x_test)
# # outlier_train = y_pred_train == -1
# # outlier_test = y_pred_test == -1
# # n_error_train = y_pred_train[outlier_train].size
# # n_error_test = y_pred_test[outlier_test].size
# 
# # plot the line, the points, and the nearest vectors to the plane
# Z_b = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
# Z_b = Z_b.reshape(xx.shape)
#
# ax.plot(
#     x_train.amplitude, x_train.decay, '.', markersize=1, zorder=2,
#     rasterized=True, label='Control')
# ax.plot(
#     x_test.amplitude, x_test.decay, '.', markersize=1, zorder=1,
#     rasterized=True, label='Perturbed')
# 
# ax.axvline(
#     x_train.mean()[0], linestyle=':', color='gray', linewidth=0.5,
#     zorder=2)
# ax.axhline(
#     x_train.mean()[1], linestyle=':', color='gray', linewidth=0.5,
#     zorder=2)
# 
# a = ax.contour(xx, yy, Z_b, levels=[0], linewidths=1, colors='k',
#                 label='Decision Boundary', zorder=3, linestyles='--')
# for c in a.collections:
#     c.set_dashes([(0, (3.0, 3.0))])
# 
# ax.plot(x_train.mean().amplitude, x_train.mean().decay, 'go',
#         markeredgecolor='white', markeredgewidth=0.5, zorder=3,
#         label='Control Mean')
#  
 
 
 
# Start of main figure
alpha = 0.01

mainfig = plt.figure(figsize=(3.425, 3.335))
gs_top = matplotlib.gridspec.GridSpec(2,2)

gs_top.update(hspace=0.1, wspace=0.1)
# gs_bot = matplotlib.gridspec.GridSpec(1,2)

# gs_top.update(bottom=0.55)
# gs_bot.update(top=0.45)

ax_all = mainfig.add_subplot(gs_top[:])

ax_c = mainfig.add_subplot(
    gs_top[0,0], aspect=1, adjustable='box-forced')
ax_p = mainfig.add_subplot(
    gs_top[0,1], aspect=1, adjustable='box-forced', sharex=ax_c, sharey=ax_c)

ax_g = mainfig.add_subplot(
    gs_top[1,0], aspect=1, adjustable='box-forced', sharex=ax_c, sharey=ax_c)

ax_r = mainfig.add_subplot(
    gs_top[1,1], aspect=1, adjustable='box-forced', polar=True)
plt.axis('off')
 

ax_all.spines['top'].set_color('none')
ax_all.spines['bottom'].set_color('none')
ax_all.spines['left'].set_color('none')
ax_all.spines['right'].set_color('none')
ax_all.tick_params(labelcolor='w', top='off', bottom='off', left='off',
                   right='off')
 
plt.setp(ax_p.get_yticklabels(), visible=False)
plt.setp(ax_c.get_xticklabels(), visible=False)
plt.setp(ax_p.get_xticklabels(), visible=False)
 

ax_c.plot(
    x_train.amplitude, x_train.decay, '.', markersize=1.0,
    color='#377EB8', zorder=1, rasterized=True)

ax_p.plot(
    library.amplitude, library.decay, '.', markersize=1.0,
    color='#E41A1C', zorder=1, rasterized=True)

ax_g.plot(
    pert_means[pvals <= alpha].amplitude,
    pert_means[pvals <= alpha].decay,
    '.', markersize=1.0, color='0.2', zorder=1, rasterized=True)


for ax in [ax_c, ax_p, ax_g]:
    ax.axvline(control_mean.amplitude, linestyle=':', color='gray',
               linewidth=0.5, zorder=2)
    ax.axhline(control_mean.decay, linestyle=':', color='gray',
               linewidth=0.5, zorder=2)

ax_c.plot(control_mean.amplitude, control_mean.decay, 'go',
          markeredgecolor='w', markeredgewidth=0.5, zorder=3)
ax_p.plot(control_mean.amplitude, control_mean.decay, 'go',
          markeredgecolor='w', markeredgewidth=0.5, zorder=3)
ax_g.plot(control_mean.amplitude, control_mean.decay, 'go',
          markeredgecolor='w', markeredgewidth=0.5, zorder=3)

ax_c.set_xlim([-8, 8])
ax_c.set_ylim([-8, 8])

ax_all.set_xlabel(r'$\log A$ ($z_R$)')
ax_all.set_ylabel('Damping rate ($z_R$)')


ax_c.text(0.95, 0.05, 'Control', verticalalignment='bottom',
          horizontalalignment='right', transform=ax_c.transAxes, fontsize='small')
ax_p.text(0.95, 0.05, 'Perturbed', verticalalignment='bottom',
          horizontalalignment='right', transform=ax_p.transAxes, fontsize='small')
ax_g.text(0.95, 0.05, 'Averaged gene effect', verticalalignment='bottom',
          horizontalalignment='right', transform=ax_g.transAxes, fontsize='small')

# Rose diagram to show distribution of outlier points

def cartesian_to_polar(X):
    """ Convert a matrix of cartesian coordinates x, y (npoints x 2) to
    polar coordinates \\rho, \\theta (npoints x 2) """

    try: X = X.values
    except AttributeError: pass

    radii = (X**2).sum(1)
    theta = np.arctan2(X[:,1], X[:,0])
    return np.vstack([radii, theta]).T


outlier_polar = cartesian_to_polar(
    pert_means[pvals < alpha] - control_mean)

hist_outlier, edges_outlier = np.histogram(
    outlier_polar[:,1], bins=12, normed=True)


n_o = float(len(outlier_polar))
hist_outlier_s = np.sqrt(hist_outlier)

h_o = hist_outlier.sum()

rmax = 0.8
bottom = 0
rads = [0, np.pi/2., np.pi, 3*np.pi/2., 2*np.pi]


bars = ax_r.bar(
    edges_outlier[:-1], hist_outlier_s,
    width=(edges_outlier[1] - edges_outlier[0]), bottom=bottom)
bars_outlier = ax_r.bar(
    edges_outlier[3:6], hist_outlier_s[3:6],
    width=(edges_outlier[1] - edges_outlier[0]), bottom=bottom,
    facecolor='#E41A1C')

ax_r.plot(0, 0, 'go', markeredgecolor='white', markeredgewidth=0.5,
          zorder=4)
ax_r.set_rmax(rmax)


for rad in rads: ax_r.axvline(rad, linestyle=':', color='gray',
                            linewidth=0.5)

ax_r.text(-3*np.pi/4, rmax*0.9375,
         '${0:.1f}\\%$'.format(hist_outlier[:3].sum()*100/h_o), 
         verticalalignment='center', horizontalalignment='center',
         size='small')
ax_r.text(-np.pi/4, rmax*0.9375,
         '${0:.1f}\\%$'.format(hist_outlier[3:6].sum()*100/h_o), 
         verticalalignment='center', horizontalalignment='center',
         size='small')
ax_r.text(np.pi/4, rmax*0.9375,
         '${0:.1f}\\%$'.format(hist_outlier[6:9].sum()*100/h_o), 
         verticalalignment='center', horizontalalignment='center',
         size='small')
ax_r.text(3*np.pi/4, rmax*0.9375,
         '${0:.1f}\\%$'.format(hist_outlier[9:].sum()*100/h_o), 
         verticalalignment='center', horizontalalignment='center',
         size='small')
ax_r.text(0.95, 0.05, 'Outlier distribution', verticalalignment='bottom',
          horizontalalignment='right', transform=ax_r.transAxes, fontsize='small')

# 
# ax.legend(
#     handles[:-1] + [control, a.collections[0]],
#     labels[:-1] + ['Control Mean', 'Decision Boundary'],
#     loc='lower right', markerscale=6., numpoints=1, ncol=2)


mainfig.savefig('outliers.svg', dpi=500)
