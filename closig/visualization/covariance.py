import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_matrices(C, ax: np.ndarray = None):
    '''
        Plot the magnitude and phase of a covariance/coherence matrix
        if ax is None, a new figure is created
    '''
    plot = False
    if ax is None:
        fig, ax = plt.subplots(1, 2)
        plot = True
    else:
        assert len(ax) == 2, 'Need two axes for coherence and phase.'
    ax[0].set_title('Coherence')
    coh_im = ax[0].imshow(np.abs(C), vmin=0, vmax=1, cmap=plt.cm.viridis)

    # append colorbar axis to each axis
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(coh_im, cax=cax, orientation='vertical')

    ax[1].set_title('Phase')
    phi_im = ax[1].imshow(np.angle(C), vmin=-np.pi,
                          vmax=np.pi, cmap=plt.cm.seismic)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(phi_im, cax=cax, orientation='vertical')

    if plot:
        plt.show()
    return ax
