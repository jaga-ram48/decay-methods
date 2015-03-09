# Bunch of overrides for matplotlib settings not changeable in
# matplotlibrc. To use, 
#
# >>> from CommonFiles.PlotOptions import PlotOptions, layout_pad
# >>> PlotOptions()
# >>> *** make plots here ***
# >>> fig.tight_layout(**layout_pad)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm

from CommonFiles.mimic_alpha import colorAlpha_to_rgb

color_rotation =  ['#377EB8', '#E41A1C', '#4DAF4A', '#984EA3', '#FF7F00',
                   '#FFFF33', '#A65628', '#F781BF']


def PlotOptions(uselatex=False):

    import matplotlib
    import matplotlib.axis, matplotlib.scale 
    from matplotlib.ticker import (MaxNLocator, NullLocator,
                                   NullFormatter, ScalarFormatter)

    MaxNLocator.default_params['nbins']=6
    MaxNLocator.default_params['steps']=[1, 2, 5, 10]

    def set_my_locators_and_formatters(self, axis):
        # choose the default locator and additional parameters
        if isinstance(axis, matplotlib.axis.XAxis):
            axis.set_major_locator(MaxNLocator())
        elif isinstance(axis, matplotlib.axis.YAxis):
            axis.set_major_locator(MaxNLocator())
        # copy&paste from the original method
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())
    #override original method
    matplotlib.scale.LinearScale.set_default_locators_and_formatters = \
            set_my_locators_and_formatters

    matplotlib.backend_bases.GraphicsContextBase.dashd = {
            'solid': (None, None),
            'dashed': (0, (2.0, 2.0)),
            'dashdot': (0, (1.5, 2.5, 0.5, 2.5)),
            'dotted': (0, (0.25, 1.50)),
        }

    matplotlib.colors.ColorConverter.colors['f'] = \
            (0.3058823529411765, 0.00784313725490196,
             0.7450980392156863)
    matplotlib.colors.ColorConverter.colors['h'] = \
            (0.5843137254901961, 0.0, 0.11372549019607843)
    matplotlib.colors.ColorConverter.colors['i'] = \
            (0.00784313725490196, 0.2549019607843137,
             0.0392156862745098)
    matplotlib.colors.ColorConverter.colors['j'] = \
            (0.7647058823529411, 0.34509803921568627, 0.0)

    if uselatex:
        matplotlib.rc('text', usetex=True)
        matplotlib.rc('font', family='serif')
        

# Padding for formatting figures using tight_layout
layout_pad = {
    'pad'   : 0.05,
    'h_pad' : 0.6,
    'w_pad' : 0.6}

# Plot shortcuts for a number of circadian-relevant features

def plot_gray_zero(ax, **kwargs):
    ax.axhline(0, ls='--', color='grey', **kwargs)

class HistRCToggle:
    """ Class to toggle the xtick directional update of
    histogram-specific RC settings """

    hist_params = {'xtick.direction' : 'out',
                   'ytick.direction' : 'out'}

    def __init__(self):
        self.rcdef = plt.rcParams.copy()

    def on(self):
        plt.rcParams.update(self.hist_params)

    def off(self):
        plt.rcParams.update(self.rcdef)


def histogram(ax, data1=None, data2=None, color1=None, color2=None,
              bins=20, range=None, label1=None, label2=None,
              normed=True, r_format=True, legend=False):
    """ Function to display a pretty histogram of up to two different
    distributions. Approximates transparency to get around annoying
    issue of pdflatex and matplotlib. """

    if color1 == None: color1 = color_rotation[0]
    if color2 == None: color2 = color_rotation[1]

    try:
        # Deal with pandas dataframes
        data1 = data1.values
        data2 = data2.values
    except AttributeError: pass
    
    # c1_on_w = colorAlpha_to_rgb(color1, alpha=0.5, bg='w')[0]
    hist1 = ax.hist(
        data1, range=range, bins=bins, facecolor=color1,
        edgecolor='white', normed=normed)
    if range:
        ax.set_xlim(range)

    if data2 is not None:
        c2_on_w = colorAlpha_to_rgb(color2, alpha=0.5, bg='w')[0]
        c2_on_c1 = colorAlpha_to_rgb(color2, alpha=0.5, bg=color1)[0]

        hist2 = ax.hist(
            data2, range=range, bins=bins, normed=normed,
            facecolor=c2_on_w, edgecolor='white') 

        # ax.legend(loc='upper left')

        orders = hist2[0] > hist1[0]
        for i, order in enumerate(orders):
            if order:
                hist1[-1][i].set_facecolor(c2_on_c1)
                hist1[-1][i].set_zorder(2)
            else:
                hist2[-1][i].set_facecolor(c2_on_c1)

    if r_format:
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['bottom'].set_position(('axes', -0.025))
        ax.yaxis.set_ticks_position('left')
        ax.spines['left'].set_position(('axes', -0.025))
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)

    if legend:
        patch1 = matplotlib.patches.Patch(color=color1, label=label1)
        patch2 = matplotlib.patches.Patch(color=c2_on_w, label=label2)
        ax.legend(handles=[patch1, patch2], loc=legend)


def density_contour(ax, xdata, ydata, nbins_x=30, nbins_y=None,
                    xy_range=None, levels=None, cmap=None,
                    **contour_kwargs):
    """ Create a density contour plot.

    Parameters
    ----------
    xdata : numpy.ndarray
    ydata : numpy.ndarray
    nbins_x : int
        Number of bins along x dimension
    nbins_y : int
        Number of bins along y dimension (default: nbins_x)
    ax : matplotlib.Axes (optional)
        If supplied, plot the contour to this axis. Otherwise, open a
        new figure
    contour_kwargs : dict
        kwargs to be passed to pyplot.contour()
    """

    if cmap == None: cmap = matplotlib.cm.PuBu
    if cmap == "none": cmap = None

    if nbins_y == None: nbins_y = nbins_x

    H, xedges, yedges = np.histogram2d(xdata, ydata,
                                       bins=(nbins_x, nbins_y),
                                       normed=True, range=xy_range)

    pdf = H

    X, Y = 0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1])
    Z = pdf.T

    contour = ax.contour(X, Y, Z, levels=levels, colors='0.2')
    ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=LogNorm()
                **contour_kwargs)

    return contour

def lighten_color(color, degree):
    cin = matplotlib.colors.colorConverter.to_rgb(color)
    cw = np.array([1.0]*3)
    return tuple(cin + (cw - cin)*degree)

def color_range(NUM_COLORS, cm=None):
    if cm is None: cm = matplotlib.cm.get_cmap('gist_rainbow')
    return (cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS))


solarized = {
    'yellow'  : '#b58900',
    'orange'  : '#cb4b16',
    'red'     : '#dc322f',
    'magenta' : '#d33682',
    'violet'  : '#6c71c4',
    'blue'    : '#268bd2',
    'cyan'    : '#2aa198',
    'green'   : '#859900',
}

def boxplot(ax, data, color='#377EB8', sym='b.'):
    """ Create a nice-looking boxplot with the data in data. Columns
    should be the different samples. sym handles the outlier mark,
    default is no mark. """

    data = np.asarray(data)

    # Shortcut method if there is no nan data
    if not np.any(np.isnan(data)): cdata = data
    else:
        cdata = [col[~np.isnan(col)] for col in data.T]

    bp = ax.boxplot(cdata, sym=sym, widths=0.65)
    plt.setp(bp['medians'], color=color, linewidth=0.75,
             solid_capstyle='butt')
    plt.setp(bp['boxes'], color=color, linewidth=0.5)
    plt.setp(bp['whiskers'], color=color, linewidth=0.5, linestyle='--',
             dashes=(4,3))
    plt.setp(bp['caps'], color=color, linewidth=0.5)
    plt.setp(bp['fliers'], markerfacecolor=color)


def barplot(ax, data, labels):
    """ create a nice barplot, where data is a 1-d array, and labels are
    the IDs of each of the elements """

    N = len(data)
    ax.bar(np.arange(N) - 0.5, data, width=0.9)
    ax.set_xlim([-0.6, N - 0.4])
    ax.set_xticks(np.arange(N))
    ax.set_xticklabels(labels)
