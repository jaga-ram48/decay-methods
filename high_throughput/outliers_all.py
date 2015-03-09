import numpy as np

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
    :, ['amplitude', 'decay', 'period', 'phase', 'Gene_ID']]
control = prune_outliers(zhang.scaled_blfit_c, max_dist=8)
# means = library.groupby('Gene_ID').mean()

from sklearn.covariance import EllipticEnvelope, MinCovDet
from scipy.stats import chi2

x_train = control.loc[:, ['amplitude', 'decay', 'period', 'phase']]

S = MinCovDet().fit(x_train)
control_mean = S.location_
inv_cov = np.linalg.pinv(S.covariance_)

# control_mean = x_train.mean()
# inv_cov = np.linalg.pinv(x_train.cov())


def hotelling_tsquared(x):
    """ Function to test the perturbed population x (all with the same
    Gene ID) against the control population in x_test, assuming equal
    covariances """

    pert_mean = x.drop(['Gene_ID'], 1).mean()
    mean_diff = control_mean - pert_mean
    T2 = mean_diff.dot(len(x)*inv_cov).dot(mean_diff)
    return 1 - chi2.cdf(T2, 2)

pvals = library.groupby('Gene_ID').apply(hotelling_tsquared)
pert_means = library.groupby('Gene_ID').mean()


alpha = 0.01

print "        Total genes considered: {0:d}".format(len(pvals))
print "  Number of insignificant hits: {0:d}".format(sum(pvals > alpha))
print "    Number of significant hits: {0:d}".format(sum(pvals <= alpha))
print "Percentage of significant hits: {0:0.2f}".format(100*float(sum(pvals <= alpha))/len(pvals))

