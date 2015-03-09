# This file contains a collection of algorithms to smooth, detrend, and
# fit an exponentially decaying sinusoidal signal.
# The main function is fit_data(x, sampling_period), which returns a
# series object containing the relevant fitted parameters. Fitted
# trajectories can also be returned by calling 
# fit_data(x, sampling_period, full=True)


import pandas as pd
import numpy as np
from scipy import linalg

import pymc as pm



def fit_data(x, sampling_period, full=False, outliers=False):
    """ Function to detrend, denoise, and fit an exponential sinsoid to
    the data in x. """

    # Container to hold output values
    return_dict = {}
    if full: return_dict['sampling_period'] = sampling_period

    x = np.asarray(x, float)

    # If any data points are missing (at the end), prune the dataset to
    # only include the finite values
    inds = np.isnan(x)
    if np.any(inds):
        x = x[:inds.tolist().index(True)]
    if full: return_dict['x_raw'] = np.array(x)

    # Pop outliers if the flag is given
    if outliers: 
        try:
            x = outlier_removal(x, sampling_period)
            if full: return_dict['x_kalman'] = np.array(x)

        except RuntimeError:
            # If the outlier removal algorithm fails, return a blank
            # series
            return_dict.update({
                'period'          : np.nan,
                'decay'           : np.nan,
                'amplitude'       : np.nan,
                'phase'           : np.nan,
                'R2'              : 0.,
            })
            return pd.Series(return_dict)


    # Find the relevant lambda parameter for the Hodrick-Prescott
    # Filter. Power law scaling as recommended by Ravn, Uhlig 2004, with
    # scaling calculated empirically as 0.05
    points_per_period = 24./sampling_period
    w = min(0.05*points_per_period**4, 10000)

    # Apply the hodrick-prescott filter to remove non-circadian
    # drift and center around 0
    x = hpfilter(x, w)
    if full: return_dict['x_detrend'] = np.array(x)

    # Use a low-pass filter to remove high-noise components (anything
    # with a period less than 4hrs).
    cutoff = max(sampling_period/2., 0.1)
    x = lowpass_filter(x, cutoff)
    if full: return_dict['x_lowpass'] = np.array(x)

    # Fit the frequency and decay parameters
    Om, D = fMatPen(x)

    # Fit the amplitude and phase parameters
    amp, phase = ls_amp_phase(x, Om, D)

    # Calculate predicted sinusoid
    n = np.arange(len(x))
    x_hat = amp*np.exp(-D*n)*np.cos(Om*n + phase)
    if full: return_dict['x_hat'] = x_hat

    # Calculate R2 of the fit
    r2 = 1 - ((x - x_hat)**2).sum() / ((x - x.mean())**2).sum()

    # Scale period, decay
    period = float((2*np.pi/Om)*sampling_period)
    decay  = float(D/sampling_period)

    return_dict.update({
        'period'          : period,
        'decay'           : decay,
        'amplitude'       : amp,
        'phase'           : phase,
        'R2'              : r2,
    })

    return pd.Series(return_dict)


def fMatPen(x, K=1):
    """ Estimate the periods and decays of the given signal using the
    using the linear-prediction SVD-based Matrix Pencil method. K is the
    assumed number of summed sinusoids.

    Method from: 
    1. Y. Hua, T. K. Sarkar, IEEE Trans. Acoust. 38, 814-824 (1990).
    2. T. K. Sarkar, O. Pereira, IEEE Antennas Propag. Mag. 37, 48-55
       (1995). 

    Code adapted from matlab routines from:
    3. T. Zielinski, K. Duda, Metrol. Meas. Syst. 18, 505-528 (2011).
    """
    M = 2*K
    N = len(x)
    L = int(N/3)
    X = linalg.hankel(x[:N-L], x[N-L-1:])
    U, S, V = linalg.svd(X[:,1:L+1], full_matrices=False)
    V = V.T
    P = np.diag(1./S[:M]).dot((U[:,:M].T.dot(X[:,:L])).dot(V[:,:M]))
    p = np.log(linalg.eigvals(P))
    Om = np.imag(p)
    indx = Om.argsort()[::-1]
    Om = Om[indx[:K]]
    D = np.real(p[indx[:K]])

    return Om, D

def ls_amp_phase(x, Om, D):
    """ Use a linear least-squares solution to estimate the remaining
    sinusoidal parameters (amplitude and phase) from a given signal and
    previously-estimated decay and freq. 

    Method from:
    1. T. Zielinski, K. Duda, Metrol. Meas. Syst. 18, 505-528 (2011).
    """

    n = np.arange(len(x))
    E1 = np.exp(1j*(Om + 1j*D)*n)
    E2 = np.exp(-1j*(Om - 1j*D)*n)
    E = np.vstack([E1, E2]).T
    c = np.linalg.lstsq(E, x)[0]
    amplitude = 2*np.abs(c[0])
    phase = np.angle(c[0])

    
    return amplitude, phase

from scipy.sparse import dia_matrix, eye as speye
from scipy.sparse.linalg import spsolve
def hpfilter(x, lamb):
    """ Code to implement a Hodrick-Prescott with smoothing parameter
    lambda. Code taken from statsmodels python package (easier than
    importing/installing, https://github.com/statsmodels/statsmodels).
    Returns the detrended signal """

    x = np.asarray(x, float)
    if x.ndim > 1:
        x = x.squeeze()
    nobs = len(x)
    I = speye(nobs,nobs)
    offsets = np.array([0,1,2])
    data = np.repeat([[1.],[-2.],[1.]], nobs, axis=1)
    K = dia_matrix((data, offsets), shape=(nobs-2,nobs))

    trend = spsolve(I+lamb*K.T.dot(K), x, use_umfpack=True)

    return x-trend

from scipy import signal
def lowpass_filter(x, cutoff=0.4, order=5):
    """ Filter the data in x with a forward-backward lowpass filter.
    Cutoff specifies the critical frequency of the butterworth filter in
    fraction of the nyquist frequency. Lower cutoffs for more filtering.
    """

    x = np.asarray(x, float)
    b, a = signal.butter(order, cutoff)
    x_filt = signal.filtfilt(b, a, x)

    return x_filt

from methods.AccOutlierFilter import AccOutlierFilter
def outlier_removal(x, sampling_period):
    """ Use a kalman filter to remove outliers from the dataset and
    smooth the resulting trajectory. Assumes outliers are not present in
    the first 25 data points. """

    filt = AccOutlierFilter(x, sampling_period).run_filter()
    filt.recalibrate()
    filt.run_filter()
    return filt.smooth()[:,0]


if __name__ == "__main__":
    # The following code tests the algorithm if it is called directly

    lumin = pd.read_csv('/lumin.csv')
    names = pd.read_csv('/names.csv')

    sp1 = 0.023
# 
#     x1 = np.array([
#        100774.,   84304.,   73794.,   69194.,   55889.,   45772.,
#         41533.,   36665.,   32828.,   29662.,   27836.,   26166.,
#         25268.,   25629.,   25660.,   25815.,   26764.,   27496.,
#         27826.,   29383.,   29270.,   30920.,   31085.,   31828.,
#         32344.,   33406.,   35757.,   36727.,   37181.,   38006.,
#         38893.,   39470.,   39542.,   39336.,   40429.,   40821.,
#         40058.,   38594.,   39419.,   38202.,   37511.,   35520.,
#         34148.,   33829.,   32797.,   30827.,   29920.,   28517.,
#         28012.,   26877.,   26496.,   26155.,   25609.,   24784.,
#         24412.,   24206.,   24629.,   24474.,   24742.,   25794.,
#         26815.,   27341.,   28218.,   29136.,   29879.,   31240.,
#         32539.,   34097.,   35283.,   35304.,   33076.,   38501.,
#         39140.,   40863.,   41028.,   40986.,   42523.,   43616.,
#         43379.,   43224.,   42276.,   43750.,   42337.,   42141.,
#         42812.,   42203.,   41347.,   41038.,   40718.,   40089.,
#         40130.,   40110.,   39821.,   39078.,   38944.,   38645.,
#         38583.,   38243.,   38140.,   38903.,   38532.,   38697.,
#         38975.,   39192.,   40347.,   40605.,   41358.,   42729.,
#         42018.,   43028.,   43854.,   44266.,   45431.,   45576.,
#         46514.,   47412.,   48031.,   48041.,   48484.,   49505.,
#         49557.,   49248.,   50341.,   50485.,   50960.,   49495.,
#         50093.,   50289.,   49650.,   50671.,   51795.,   52249.,
#         51166.,   51795.,   51310.,   51403.,   51981.,   52569.,
#         52465.,   52527.,   51929.,   52414.,   52465.,   52960.,
#         53074.,   53486.,   54095.,   54703.,   54683.,   54868.,
#         54920.,   54023.,   56075.,   56828.,   57839.,   57045.,
#         57859.,   57870.,   58489.,   58839.,   59241.,   60613.,
#         61077.,   60758.,   57488.,   61892.,   61871.,   61510.,
#         63047.,   63872.,   64594.,   64099.,   63841.,   66296.,
#         64852.,   65852.,   61376.,   57344.,   57767.,   60778.,
#         65481.,   67874.,   61294.,   70916.,   72247.,   72567.,
#         68792.,   71267.,   71133.,   71246.,   71525.,   70421.,
#         71401.,   72618.,   73134.,   72866.,   72928.,   73144.,
#         73546.,   73660.,   74217.,   74640.,   75032.,   75104.,
#         75547.,   75269.,   75475.,   75599.,   76785.,   77393.,
#         76125.,   77001.,   76919.,   77579.,   78002.,   77424.,
#         79085.,   79724.,   80219.,   78157.,   80281.,   81055.,
#         79838.,   81952.,   82643.,   82736.,   82901.,   82457.,
#         83437.,   83499.,   83272.,   85851.,   86232.,   84902.,
#         87109.,   85221.,   84407.,   85490.,   87243.,   87521.,
#         88089.,   87985.,   88233.,   88274.,   88192.,   86768.,
#         88656.,   89749.,   89553.,   88243.,   89450.,   88584.,
#         89027.,   89574.,   89759.,   91348.,   91874.,   91203.,
#         90801.,   90615.,   91482.,   90481.,   91245.,   93648.,
#         93534.,   93318.,   92523.,   92255.,   94040.,   92843.,
#         93431.,   94844.,   94493.,   94462.,   95556.,   94277.,
#         94473.,   94174.,   96752.,   95721.,   93957.,   85221.,
#         74248.,   63037.,   58509.,   59974.,   55982.,   72391.,
#        100094.,  104333.,  105869.,  106447.,  103961.,  103559.,
#        103652.,  103054.,  101496.,  100733.,   99021.,   99681.,
#         98083.,   98588.])
#     

    x1 = np.array([ 111.,   83.,   75.,   80.,   94.,  122.,  141.,  164.,
                194., 215.,  232.,  215.,  189.,  154.,  118.,   91.,
                70.,   61., 52.,   49.,   50.,   59.,   69.,   78.,
                91.,  110.,  117., 116.,  122.,  117.,  109.,  102.,
                94.,   77.,   76.,   72., 68.,   72.,   72.,   81.,
                86.,   88.,   93.,   97.,  100., 96.,   95.,   92.,
                94.,   86.,   76.,   81.,   74.,   77., 76.,   73.,
                79.,   76.,   82.,   84.,   81.,   86.,   87., 83.,
                82.,   76.,   76.,   76.,   74.,   67.,   69.,   66.])
    sp1 = 1.66


    # sp1 = 2.
    # x1 = np.array([ -93.376, -128.174, -115.173,  -46.591,   35.161,
    #               92.173,  133.255,  141.447,  133.079,   68.621,
    #               11.983,  -32.145,  -57.393,  -68.721,  -60.759,
    #               -44.467,   -9.305,   21.757,   49.879,   64.551,
    #               52.803,   33.005,    2.747,  -16.561,  -37.12 ,
    #               -48.248,  -48.866,  -44.504,  -28.652,  -15.72 ,
    #               -2.928,   11.604,   16.506,   16.048,   10.81 ,
    #               3.272])


    out = fit_data(x1[3:], sp1, full=True, outliers=True)
    ts = np.arange(len(out['x_raw'])) * out['sampling_period']



    import matplotlib.pyplot as plt

    fig, axmatrix = plt.subplots(nrows=2, sharex=True)

    ax1 = axmatrix[0]
    ax2 = axmatrix[1]

    ax1.plot(ts, out['x_raw'], '-', label="Experimental Data")
    ax1.plot(ts, out['x_kalman'], '-', label="Outliers removed")
    ax1.plot(ts, out['x_kalman'] - out['x_detrend'], '--', label="Mean")
    ax1.legend(loc='upper right')


    ax2.plot(ts, out['x_detrend'], '-', label='Detrended')
    ax2.plot(ts, out['x_lowpass'], '-', label='Lowpass Filter')
    ax2.plot(ts, out['x_hat'], '-', label='Fitted sinusoid')
    ax2.legend(loc='upper right')




    plt.show()
