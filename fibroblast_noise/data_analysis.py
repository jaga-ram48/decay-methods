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

# We need to rank the cells by 'noisyness', which must be a single-value
# feature. We normalize the three features to zero mean and unit
# variance, and use PCA to condense the features to one value.
from sklearn.decomposition import PCA
from sklearn import preprocessing

scaled_features = preprocessing.scale(noise_features)
pca = PCA(n_components=1)
pca.fit(scaled_features)
noise_metric = pd.DataFrame(pca.transform(scaled_features),
                            index=noise_features.index, columns=['noise'])

# We next assign a rank to each noise value
noise_sort = noise_metric.sort(columns=['noise'])
noise_sort['noise_rank'] = np.arange(len(noise_sort))


# We next need to artifically synchronize the population of cells, since
# the cells in this population seem to not be synchronized. This is done
# by finding the first instance where the hillbert transform crosses
# zero, and setting the start time to that value.
endpoint = data_p.shape[0] - stats.start_ind.max()
data_s = {'days' : data_p.index.values[:endpoint]}
for name, cell in data_p.iteritems():
    offset = stats.loc[name].start_ind
    data_s[name] = cell.values[offset : (endpoint + offset)]

# Reset the index to start at 0 days
data_s = pd.DataFrame(data_s).set_index('days')
data_s.index = data_s.index.values - data_s.index[0]


# Here we join the noise rankings to the synchronized cell luminescence
# profiles.
data_s_sorted = data_s.T.join(noise_sort.iloc[:,-1:])
data_s_sorted = data_s_sorted.set_index('noise_rank').sort_index()

stats_sorted = stats.join(noise_sort.iloc[:,-1:])
stats_sorted = stats_sorted.set_index('noise_rank').sort_index()


# Next we preprocess the bioluminescence profiles to make sure the
# additions are representative
def dwt_process(series):
    """ Use a discrete wavelet transform and a hilbert transform to
    estimate the period and amplitude distributions for the cell """

    dwt_dict = dwt_breakdown(series.index, series)
    
    return dwt_dict['components'][:,:].sum(1)


data_s_scaled = data_s_sorted.apply(dwt_process, axis=1)



from methods.sinusoid_estimation import fit_data
def estimate_decaying_sinusoid(series, full=False):
    """ Function to call the decaying sinusoid estimation """

    sampling_period = (series.index[1] - series.index[0])*24.
    return fit_data(series.values, sampling_period, full=full)


# Split these in half (two groups), low-noise and high-noise
split_point = int(data_s.shape[1]/2)
t_end = 192 # 4 days

low_measured = estimate_decaying_sinusoid(
    data_s_scaled.iloc[:split_point,:t_end].mean(0), full=True)
high_measured = estimate_decaying_sinusoid(
    data_s_scaled.iloc[-split_point:,:t_end].mean(0), full=True)

test_statistics = np.abs(high_measured - low_measured).iloc[:5]


# Next we randomly assign the cells to either the high or low noise
# group, to find the expected distribution in decay rate difference.
n_rep = 10000
bootstrap_values = np.zeros((n_rep, 5))

np.random.seed(0)
sorts = np.zeros((n_rep, data_s.shape[1]))
for i in xrange(n_rep):
    index = np.arange(data_s.shape[1])
    np.random.shuffle(index)
    sorts[i] = index
    low_boot = estimate_decaying_sinusoid(
        data_s_sorted.iloc[index[:split_point],:t_end].mean(0))
    high_boot = estimate_decaying_sinusoid(
        data_s_sorted.iloc[index[-split_point:],:t_end].mean(0))
    bootstrap_values[i] = np.abs(high_boot.values - low_boot.values)

bootstrap_values = pd.DataFrame(bootstrap_values,
                                columns=low_boot.index)


p_vals = (bootstrap_values >= test_statistics).sum(0)/float(n_rep)


from methods.PlotOptions import PlotOptions, color_rotation, layout_pad
PlotOptions(uselatex=True)
import matplotlib.pylab as plt
import matplotlib


fig = plt.figure(figsize=(3.425, 3.425))
gs_top = matplotlib.gridspec.GridSpec(2,1)
gs_bot = matplotlib.gridspec.GridSpec(1,2)

gs_top.update(
    top=0.98, bottom=0.575, left=0.10, right=0.95, hspace=0.05)
gs_bot.update(
    top=0.425, left=0.10, right=0.95, wspace=0.33)

ax_all = fig.add_subplot(gs_top[:,:])
ax_all.spines['top'].set_color('none')
ax_all.spines['bottom'].set_color('none')
ax_all.spines['left'].set_color('none')
ax_all.spines['right'].set_color('none')
ax_all.tick_params(labelcolor='w', top='off', bottom='off', left='off',
                   right='off')

ax0 = fig.add_subplot(gs_top[0])
ax1 = fig.add_subplot(gs_top[1], sharex=ax0)
plt.setp(ax0.get_xticklabels(), visible=False)


ax0.plot(data_s_scaled.iloc[2:3].T, '-', linewidth=0.5,
         color=color_rotation[0])
ax1.plot(data_s_scaled.iloc[-3:-2].T, '-', linewidth=0.5,
         color=color_rotation[1])
ax1.set_xlim([0, 1200])
ax0.text(0.025, 0.925, 'Low noise', horizontalalignment='left',
                 verticalalignment='top',
                 transform=ax0.transAxes)
ax1.text(0.025, 0.925, 'High noise', horizontalalignment='left',
                 verticalalignment='top',
                 transform=ax1.transAxes)


ax0.set_ylim([-4, 9])
ax0.set_yticks([-3, 0, 3, 6, 9])

ax1.set_ylim([-3, 5])
ax1.set_yticks([-2, 0, 2, 4])

ax_all.set_xlabel('Time (hrs)')
ax_all.set_ylabel('Bioluminescence')




ax2 = fig.add_subplot(gs_bot[1])
ax2.hist(bootstrap_values.decay, bins=20, normed=True)
ax2.axvline(x=test_statistics.decay, linestyle='--', color='#E41A1C')
ax2.set_xlim([-0, 0.015])

ax2.set_ylabel('Relative Frequency')
ax2.set_xlabel('Damping Rate Difference')

ax2.text(test_statistics.decay+0.0005, 120,
         '$p={0:.4f}$'.format(p_vals.decay),
        horizontalalignment='left', verticalalignment='center')

t = np.arange(t_end)*low_measured.sampling_period
# fig = plt.figure(figsize=(2.5, 1.875))
ax3 = fig.add_subplot(gs_bot[0])
ax3.plot(t, low_measured.x_detrend, color='#377EB8', label='Low noise')
ax3.plot(t, high_measured.x_detrend, color='#E41A1C', label='High noise')

ax3.plot(t, low_measured.amplitude*np.exp(-low_measured.decay*t), '--',
        color='#377EB8')
ax3.plot(t, high_measured.amplitude*np.exp(-high_measured.decay*t), '--',
        color='#E41A1C')
ax3.plot(t, -low_measured.amplitude*np.exp(-low_measured.decay*t), '--',
        color='#377EB8')
ax3.plot(t, -high_measured.amplitude*np.exp(-high_measured.decay*t), '--',
        color='#E41A1C')

ax3.set_xlim([0, t[-1]])
ax3.set_xlabel('Time (hrs)')
ax3.set_ylabel('Bioluminescence')


ax3.set_ylim([-1, 2])
ax3.set_yticks([-1, 0, 1, 2])

ax3.legend(loc='upper right')



fig = plt.figure()
ax = fig.add_subplot(111)
x = np.arange(len(stats)) + 1
period_mu_sort = stats.period_mu.argsort().values
ax.errorbar(
    x, stats.iloc[period_mu_sort].period_mu,
    yerr=[stats.iloc[period_mu_sort].period_mu -
          stats.iloc[period_mu_sort].period_05,
          stats.iloc[period_mu_sort].period_95 -
          stats.iloc[period_mu_sort].period_mu],
    color='#262626', elinewidth=0.5, capthick=0.5, capsize=1.5,
    zorder=1, linestyle='none')

ax.plot(x, stats.iloc[period_mu_sort].period_mu, 'o',
        color='#377EB8', markersize=2, zorder=2)

ax.axhspan(stats.period_mu.quantile(0.05),
           stats.period_mu.quantile(0.95), zorder=0, color='0.85')

ax.set_ylabel('Period (days)')
ax.set_xlabel('Cell Index (sorted by mean period)')
fig.tight_layout(**layout_pad)
