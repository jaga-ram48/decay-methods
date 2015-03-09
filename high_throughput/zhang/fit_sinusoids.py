import pandas as pd
import numpy as np

from methods.Futures import map_
from methods.sinusoid_estimation import fit_data

# Load time-series data on all probes
lumin = pd.read_csv('lumin.csv')
names = pd.read_csv('names.csv')

# Prune the initial transients from the dataset.
times_considered = np.s_[2:] # Save slice for later importing
lumin = lumin.iloc[:, times_considered]

def fit(row):
    """ Wrapper function of the exponential sinusoid fitting function """
    return fit_data(row.values, names.iloc[row.name].sampling_period,
                    outliers=True)

# The following two functions allow this fitting operation to be easily
# distributed over several compute nodes using python's scoop module.
# The times data frame is sliced into sections of 100 experiments before
# being mapped onto the nodes

data_len = len(lumin)

def slice_data(section_size):
    """ Function to return a generator of indicies to slice the data
    array into smaller sections """

    ind_start = 0
    while ind_start + section_size < data_len:
        yield (ind_start, ind_start + section_size)
        ind_start += section_size
    yield (ind_start, data_len)
    
def fit_section(slice_tuple):
    fitted_section = lumin[slice(*slice_tuple)].apply(fit, axis=1)
    return fitted_section

# Placing the majority of the computational work in the following if
# statement allows the other functions to be imported without triggering
# sinsoid fitting

if __name__ == "__main__":

    results = list(map_(fit_section, slice_data(100)))
    fitted_results = pd.concat(results)
    fitted_results.to_csv('blfit.csv')
