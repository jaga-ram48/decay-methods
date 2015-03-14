import numpy as np
from methods.PlotOptions import PlotOptions, color_rotation, layout_pad
PlotOptions(uselatex=True)
import matplotlib.pylab as plt


frequencies = 1. + .02*np.random.randn(10000)
ts = np.linspace(0, 50*np.pi, 1000)
mean_ballistic = np.sin(np.outer(ts, frequencies)).mean(1)

def gauss_sin(t, decay):
    return np.exp(-decay*t**2)*np.sin(t)


from scipy.optimize import curve_fit

popt = curve_fit(gauss_sin, ts, mean_ballistic, p0=(0.05))[0]


fig, axmatrix = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True,
                             figsize=(4.83, 1.80))

axmatrix[1].plot(ts/(2*np.pi), mean_ballistic, 'b-', color='#377EB8')
axmatrix[1].plot(ts/(2*np.pi), np.exp(-popt*ts**2), '--', color='#E41A1C')
axmatrix[1].plot(ts/(2*np.pi), -np.exp(-popt*ts**2), 'r--', color='#E41A1C')


d = 0.03
axmatrix[0].plot(ts/(2*np.pi), np.exp(-d*ts)*np.sin(ts), '-',
                 color='#377EB8')
axmatrix[0].plot(ts/(2*np.pi), np.exp(-d*ts), '--', color='#E41A1C')
axmatrix[0].plot(ts/(2*np.pi), -np.exp(-d*ts), '--', color='#E41A1C')


axmatrix[1].text(18, 0.22, '$$e^{-\\tilde{d} t^2}$$',
                 verticalalignment='center',
                 horizontalalignment='center')
axmatrix[0].text(18, 0.14, '$$e^{-d t}$$',
                 verticalalignment='center',
                 horizontalalignment='center')

axmatrix[0].set_title('Cycle-to-cycle variability')
axmatrix[1].set_title('Period heterogenity')

