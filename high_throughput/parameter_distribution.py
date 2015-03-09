import numpy as np
import scipy.stats as stats
import pandas as pd

from methods.HTExperiment import Experiment

zhang = Experiment('zhang').split()


from methods.PlotOptions import (PlotOptions, histogram, HistRCToggle,
                                 layout_pad)
PlotOptions(uselatex=True)
HistToggle = HistRCToggle()

import matplotlib
import matplotlib.pylab as plt

HistToggle.on()

# Main plot of data from the zhang dataset. Differences in standard
# deviation and skew are very apparent, with the most likely value in
# each being similar

fig, axmatrix = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True,
                             figsize=(4, 1.25))

histogram(axmatrix[0], data1=zhang.scaled_blfit_c.period,
          data2=zhang.scaled_blfit_p.period, bins=20, range=(-6, 6))
histogram(axmatrix[1], data1=zhang.scaled_blfit_c.amplitude,
          data2=zhang.scaled_blfit_p.amplitude, bins=20, range=(-6, 6))
histogram(axmatrix[2], data1=zhang.scaled_blfit_c.decay,
          data2=zhang.scaled_blfit_p.decay, bins=20, range=(-6, 6),
          label1='Control', label2='Perturbed', legend='upper right')

axmatrix[0].set_title(r'Period $(z_R)$', size='medium')
axmatrix[1].set_title(r'$\log$ Amplitude $(z_R)$', size='medium')
axmatrix[2].set_title(r'Damping Rate $(z_R)$', size='medium')

axmatrix[2].legend(loc='upper right')

fig.tight_layout(**layout_pad)

def prune_outliers(df, max_dist=6, keys=None):
    """ Function to return a dataframe in which none of the elements of
    'keys' fall between -max_dist < df.key < max_dist """

    if keys == None: keys = ['decay', 'amplitude', 'period']

    return df[((df.loc[:,keys] <  max_dist) & 
               (df.loc[:,keys] > -max_dist)).all(1)].loc[:,keys]
    
def distribution_parameters(series):
    """ Function to tabulate various parameters about each distribution
    """
    return pd.Series(
        data = [np.mean(series), np.std(series), stats.skew(series),
                stats.kurtosis(series)],
        index = ['mean', 'stdv',  'skew',  'kurt']
    )


# These parameters should get placed in a table below the figure above.

control_params = prune_outliers(
    zhang.scaled_blfit_c).apply(distribution_parameters) 
perturb_params = prune_outliers(
    zhang.scaled_blfit_p).apply(distribution_parameters) 



plt.show()


# 
# 
# 
# # Similar plots for the other two datasets, although the trends are not
# # as apparent.
# 
# lopac = Experiment('lopac').split()
# maier = Experiment('maier').split()
# 
# fig, axmatrix = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True,
#                              figsize=(3.5, 2))
# 
# 
# histogram(axmatrix[0,0], data1=lopac.scaled_blfit_c.period,
#           data2=lopac.scaled_blfit_p.period, bins=20, range=(-6, 6),
#           alpha=0.5)
# histogram(axmatrix[0,1], data1=lopac.scaled_blfit_c.amplitude,
#           data2=lopac.scaled_blfit_p.amplitude, bins=20, range=(-6, 6),
#           alpha=0.5)
# histogram(axmatrix[0,2], data1=lopac.scaled_blfit_c.decay,
#           data2=lopac.scaled_blfit_p.decay, bins=20, range=(-6, 6),
#           alpha=0.5)
# 
# histogram(axmatrix[1,0], data1=maier.scaled_blfit_c.period,
#           data2=maier.scaled_blfit_p.period, bins=20, range=(-6, 6),
#           alpha=0.5)
# histogram(axmatrix[1,1], data1=maier.scaled_blfit_c.amplitude,
#           data2=maier.scaled_blfit_p.amplitude, bins=20, range=(-6, 6),
#           alpha=0.5)
# histogram(axmatrix[1,2], data1=maier.scaled_blfit_c.decay,
#           data2=maier.scaled_blfit_p.decay, bins=20, range=(-6, 6),
#           alpha=0.5)
# 
# axmatrix[0,0].set_title(r'Period', size='medium')
# axmatrix[0,1].set_title(r'Amplitude', size='medium')
# axmatrix[0,2].set_title(r'Damping Rate', size='medium')
# 
# for ax in axmatrix[:1, :].flat:
#     plt.setp(ax.get_xticklabels(), visible=False)
# 
# 
# fig.tight_layout(**layout_pad)
