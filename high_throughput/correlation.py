import numpy as np
import scipy.stats as stats
import pandas as pd

from methods.HTExperiment import Experiment

zhang = Experiment('zhang').split()

def prune_outliers(df, max_dist=6, keys=None):
    """ Function to return a dataframe in which none of the elements of
    'keys' fall between -max_dist < df.key < max_dist """

    if keys == None: keys = ['decay', 'amplitude', 'period']

    return df[((df.loc[:,keys] <  max_dist) & 
               (df.loc[:,keys] > -max_dist)).all(1)].loc[:,keys]

# Here I do a linear model to show that variation in decay is not
# explained by variation in period or amplitude
import statsmodels.api as sm


regression = prune_outliers(
    zhang.scaled_blfit, max_dist=8,
    keys=['decay', 'amplitude', 'period', 'phase'])

correlation = regression.corr()

decay = regression.pop('decay')

# Add a categorical variable
regression = regression.join(
    (zhang.names.type == 'perturbed').astype(float))

# Add an intercept
regression = sm.add_constant(regression)
model = sm.OLS(decay, regression)
results = model.fit()

# wow - r2 is only 0.169


