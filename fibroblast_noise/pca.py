import numpy as np
import pandas as pd
from scipy import signal

import pywt


def estimate_noise_and_phase(series):
    """ Use a discrete wavelet transform and a hilbert transform to
    estimate the period and amplitude distributions for the cell """

    dwt_dict = dwt_breakdown(series.index, series)

    # Leise et. al., 2012 use two dwt's, one with higher-level noise
    # included (8hr - 64hr bins) and one with less noise (16hr - 64hr
    # bins)
    
    filtered = dwt_dict['components'][:,3:-2].sum(1)

    # Find the zero-crossings of the hilbert transform phase.
    zero_crossings = np.where(np.diff(np.sign(np.angle(
        signal.hilbert(filtered)))) == 2)[0]

    periods = np.diff(series.index[zero_crossings].values)
    amplitudes = np.abs(signal.hilbert(filtered))[zero_crossings]

    return pd.Series({
        'period_mu'    : periods.mean(),
        'period_cv'    : periods.std()/periods.mean(),
        'amplitude_mu' : amplitudes.mean(),
        'amplitude_cv' : amplitudes.std()/amplitudes.mean(),
        'start_ind'    : zero_crossings[0],
        'noise'        : (dwt_dict['components'][:,:3].sum(1).std() /
                          amplitudes.mean()),
        'period_05'    : np.percentile(periods, 5),
        'period_95'    : np.percentile(periods, 95),
    })


def dwt_breakdown(x, y, wavelet='dmey', nbins=np.inf, mode='sym'):
    """ Function to break down the data in y into multiple frequency
    components using the discrete wavelet transform """

    lenx = len(x)

    # Restrict to the maximum allowable number of bins
    if lenx < 2**nbins: nbins = int(np.floor(np.log(len(x))/np.log(2)))

    dx = x[1] - x[0]
    period_bins = [(2**j*dx, 2**(j+1)*dx) for j in xrange(1,nbins+1)]

    details = pywt.wavedec(y, wavelet, mode, level=nbins)
    cA = details[0]
    cD = details[1:][::-1]

    # Recover the individual components from the wavelet details
    rec_d = []
    for i, coeff in enumerate(cD):
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, wavelet)[:lenx])

    rec_a = pywt.waverec([cA] + [None]*len(cD), wavelet)[:lenx]

    return {
        'period_bins'   : np.array(period_bins),
        'components'    : np.array(rec_d).T,
        'approximation' : np.array(rec_a).T,
    }


# Create a dataframe to hold the cell data
data = pd.read_csv('welsh_fibroblast_data.csv').set_index('days')

# Some of the cells have a shorter recording time than others. We
# therefore prune the dataset to this length to remove all the nan
# values.
def first_nan(x):
    try: return np.where(np.isnan(x) == True)[0][0]
    except IndexError: return len(x) # All values present

# Count the number of continuous non-nan samples
data_len = data.apply(first_nan)

# Restrict the dataset to only those cells which have more than 2012
# samples, cut the rest to 2012 for uniformity
data_p = data.T[data_len >= 1679].T.iloc[150:1679]


# Here we gather information of how noisy each cell is. We use three
# features, aplitude and period constant of variation (as in the
# previous study) as well as noise, estimated from the first three
# octaves of the DWT

stats = data_p.apply(estimate_noise_and_phase).T
noise_features = stats.loc[:,['amplitude_cv', 'period_cv', 'noise']]

from sklearn.decomposition import PCA
from sklearn import preprocessing
scaled_features = preprocessing.scale(noise_features)
pca = PCA(n_components=1)
pca.fit(scaled_features)
noise_metric = pd.DataFrame(pca.transform(scaled_features),
                            index=noise_features.index, columns=['noise'])



t = np.linspace(noise_metric.min().values,
                noise_metric.max().values, 100)
def line3d(t, slope, intercept):
    return (slope.T*t).T + intercept
pca_line = line3d(t, pca.components_, pca.mean_)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(scaled_features[:,0], scaled_features[:,1],
           scaled_features[:,2], s=4.0)
ax.plot(pca_line[:,0], pca_line[:,1], pca_line[:,2])

for point in scaled_features:
    pca_point = pca.inverse_transform(pca.transform(point))
    line = line3d(np.linspace(0, 1), point - pca_point, pca_point)
    ax.plot(line[:,0], line[:,1], line[:,2], color='gray', alpha=0.5)


ax.set_xlabel('Amplitude CV')
ax.set_ylabel('Period CV')
ax.set_zlabel('High Frequency Noise')

