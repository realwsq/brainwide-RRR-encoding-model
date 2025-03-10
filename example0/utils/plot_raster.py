import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering

def raster_plot(ts_, vmax, vmin, whether_cbar, title, ax, cmap='bwr'):
    """
    Plots a raster plot for 2D data with time on the x-axis and trials on the y-axis.

    Args:
        ts_ (ndarray): Time-series data to plot [K, T].
        vmax (float): Maximum value for color scale.
        vmin (float): Minimum value for color scale.
        whether_cbar (bool): Whether to display the color bar.
        title (str): Title for the plot.
        ax (matplotlib.Axes): Axes object to plot on.
        cmap (str): Colormap to use.

    Returns:
        None
    """
    # Transpose to make time on the x-axis and trials on the y-axis
    im = ax.imshow(ts_, aspect='auto', cmap=cmap, vmax=vmax, vmin=vmin, interpolation=None)

    if whether_cbar:
        cbar = plt.colorbar(im, ax=ax, pad=0.005, shrink=0.8)
        cbar.ax.tick_params(rotation=90)
    ax.tick_params(axis='both', which='major')
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel('Time')
    ax.set_ylabel('Trial')
    ax.set_title(title)



"""
:X: [n_trials, n_timesteps, n_variables]
:y: [n_trials, n_timesteps] (in Hz)
:y_pred: [n_trials, n_timesteps] (in Hz)
:nclus, n_neighbors, clusby: hyperparameters for spectral_clustering
:axes, fname: parameters used for plotting
"""
def plot_single_neuron_activity(X, y, y_pred,
                               n_clus=8, n_neighbors=5, clusby='y_pred',
                               axes=None, fname=None):
    if axes is None:
        ncols = 1; nrows = 2 + 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 3 * nrows), constrained_layout=True)
    
    if clusby is None:
        k_sort = np.arange(y.shape[0])
    else:
        clustering = SpectralClustering(n_clusters=n_clus, n_neighbors=n_neighbors,
                                            affinity='nearest_neighbors',
                                            assign_labels='discretize',
                                            random_state=0)
        if clusby == 'y_pred':
            clustering = clustering.fit(y_pred)
        elif clusby == 'y':
            clustering = clustering.fit(y)
        else:
            assert False, "invalid clusby"
        k_sort = np.argsort(clustering.labels_)
        

    toshows = [y, y_pred, y-y_pred]
    titles = [f"obs. act.", f"pred. act.", "residual act."]
    for ri in range(nrows):   
        ax = axes[ri]
        if ri <= 1:
            # plot obs./ predicted activity
            vmax = np.percentile(np.abs(y), 90)
            raster_plot(toshows[ri][k_sort], vmax, 0, True, 
                        titles[ri], ax,
                        cmap='Greys')
            ax.set_xticks([])
            ax.set_xlabel('')

        elif ri == nrows-1:
            # plot residual activity
            vmax = np.percentile(np.abs(toshows[-1]), 90)
            vmin = -vmax
            raster_plot(toshows[-1][k_sort], vmax, vmin, True, 
                        titles[-1], ax,
                        cmap='bwr',)

    if fname is not None:
        plt.savefig(fname)