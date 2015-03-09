import pandas as pd
import numpy as np

import matplotlib.pylab as plt
from CommonFiles.PlotOptions import (PlotOptions, solarized, layout_pad,
                                     histogram, HistRCToggle, blue,
                                     lighten_color)

class Experiment(object):
    """ A class to hold the experiment-specific data, which will
    hopefully make comparisons between the various sets easier to make
    """

    def __init__(self, name, threshold=0.8):
        """ Load data stored in the directory specified folder """

        self.folder = name
        self.raw_lumin = pd.read_csv(name + '/lumin.csv')
        self.raw_names = pd.read_csv(name + '/names.csv')
        self.raw_blfit = pd.read_csv(name + '/blfit.csv')

        # Rescale the amplitude to a log-scale
        self.raw_blfit.amplitude = np.log(self.raw_blfit.amplitude)

        # Only keep results with a high R2 fit value
        high_r2 = self.raw_blfit.R2 > threshold
        self.lumin = self.raw_lumin[high_r2].copy()
        self.names = self.raw_names[high_r2].copy()
        self.blfit = self.raw_blfit[high_r2].copy()

        # Rescale the phase to be +/- from the mean phase
        mean_phase = np.angle(np.exp(-1j*self.blfit.phase).mean())
        self.blfit.phase += mean_phase
        self.blfit.loc[self.blfit.phase < -np.pi, 'phase'] += 2*np.pi
        self.blfit.loc[self.blfit.phase >  np.pi, 'phase'] -= 2*np.pi

        # To protect against plate-to-plate variation, we find a robust
        # z-score statistic for each parameter, normalized on a
        # plate-by-plate basis.
        def robust_z(x):
            x = x.drop(['plate'], axis=1)
            dev = x - x.median()
            mad = np.abs(dev).median()
            return dev / mad

        annotated_fits = self.blfit.join(self.names.loc[:, ['plate']])
        self.scaled_blfit = (
            annotated_fits.groupby('plate').apply(robust_z))


    def split(self, center_strategy='none'):
        control = self.names['type'] == 'control'
        perturb = self.names['type'] == 'perturbed'

        self.lumin_c = self.lumin[control] 
        self.names_c = self.names[control] 
        self.blfit_c = self.blfit[control] 
        self.scaled_blfit_c = self.scaled_blfit[control] 

        self.lumin_p = self.lumin[perturb] 
        self.names_p = self.names[perturb] 
        self.blfit_p = self.blfit[perturb] 
        self.scaled_blfit_p = self.scaled_blfit[perturb] 

        # Renormalize the center point of the scaled distributions

        if center_strategy == 'none': pass
        
        elif center_strategy == 'control':
            center = self.scaled_blfit_c.median()
            self.scaled_blfit_c -= center
            self.scaled_blfit_p -= center

        elif center_strategy == 'each':
            # For the hirota dataset, the control median was
            # significantly different than that of the small molecule
            # perturbations, which was likely due to systemic effects.
            # Regardless, the distribution of control points gives us an
            # idea of the accuracy of the measurement/fitting procedure,
            # so finding outliers from seperately centered data is still
            # a valid approach
            center_c = self.scaled_blfit_c.median()
            center_p = self.scaled_blfit_p.median()
            self.scaled_blfit_c -= center_c
            self.scaled_blfit_p -= center_p

        self.num_plates = len(self.names.plate.unique())
        self.nc = len(self.blfit_c)
        self.np = len(self.blfit_p)

        return self

    def plate_to_plate_variation(self):
        """ Function to check the variation between fitted values in
        each plate. """

        annotated_fits = self.blfit.join(
            self.names.loc[:, ['plate', 'well']])

        def box(ax, param, color):
            pivot = annotated_fits.pivot(index='well', columns='plate',
                                          values=param)
            
            if self.num_plates < 40:

                bp = ax.boxplot(pivot.values, notch=False, sym='',
                                widths=0.65, whis=[2.5, 97.5])
                plt.setp(bp['medians'], color=color, linewidth=0.75,
                         solid_capstyle='butt')
                plt.setp(bp['boxes'], color=color, linewidth=0.5)
                plt.setp(bp['whiskers'], color=color, linewidth=0.5)
                plt.setp(bp['caps'], color=color, linewidth=0.5)
                plt.setp(bp['fliers'], color=color)

            else:
                # Just draw a line from .25 to 0.75 quartile, with dot
                # for median 
                pass
                lower = pivot.quantile(0.25)
                upper = pivot.quantile(0.75)
                for i, (l,u) in enumerate(zip(lower, upper)):
                    ax.plot([i+1, i+1], [l, u],
                            color=lighten_color(color, 0.75), zorder=0)
                ax.plot(np.arange(self.num_plates) + 1,
                        pivot.median().values, '.', color=color, zorder=1)


            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax.set_ylabel(param)


        fig, axmatrix = plt.subplots(nrows=5, ncols=1, sharex=True)

        params = self.blfit.columns.values
        for ax, param, color in zip(axmatrix, params,
                                    solarized.values()):
            box(ax, param, color)


        if self.folder == 'zhang': 

            axmatrix[0].set_ylim([0.75, 1.0])
            axmatrix[1].set_ylim([4, 5.5])
            axmatrix[2].set_ylim([0.01, 0.04])
            axmatrix[3].set_ylim([22.5, 28])
            axmatrix[4].set_ylim([-np.pi, np.pi])

            axmatrix[4].set_yticks([-np.pi, 0, np.pi])
            axmatrix[4].set_yticklabels([r'$-\pi$', r'$0$', r'$\pi$'])

        axmatrix[4].set_xlim([0, self.num_plates+1])

        axmatrix[4].set_xlabel('Plate')
        fig.tight_layout(**layout_pad)

        return fig, axmatrix

    def check_fit(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)

        HistToggle = HistRCToggle()
        weights = 1./len(self.blfit.R2.values)
        # Plot R2 frequency of all fits
        HistToggle.on()
        hist1 = histogram(ax, self.blfit.R2[self.blfit.R2 <= 0.8].values,
                         range=(0,1.),
                         color1='gray', weights1=weights)
        hist2 = histogram(ax, self.blfit.R2[self.blfit.R2 > 0.8].values,
                         range=(0,1.),
                         weights1=weights)
        ax.set_xlabel(r'$R^2$')
        ax.set_ylabel(r'Relative Frequency')
        HistToggle.off()
    
        fig.tight_layout(**layout_pad)
        
        return fig

