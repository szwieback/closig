from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

from closig.experiments import  CutOffExperiment, model_catalog
from closig.visualization import prepare_figure, cmap_clipped, cmap_cyclic, triangle_plot
from closig import TwoHopBasis, SmallStepBasis

def speckle_plot(fnout=None):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    P = 91
    P_year = 30
    N_samples = 5
    L = 169
    xticks_year = (0, 1, 2, 3)
    model = model_catalog('decorrsoil', P_year=P_year, band='C')
    ylabels = [
        ('small step', '$\\tau$ [yr]'), ('two hop', '$ \\tau$ [yr]')]
    aspect = 0.75
    triangleparms = [
        (180, cmap_cyclic, ['$-\\pi$', '0', '$\\pi$']),
        (45, cmap_clipped, ['$-\\pi/4$', '0', '$\\pi/4$'])]
    fig, axs = prepare_figure(
        nrows=2, ncols=N_samples, figsize=(2.04, 0.63), sharey='row', sharex='col', top=0.94, left=0.07,
        wspace=0.20, hspace=0.16, bottom=0.14, right=0.90)
    for ax in axs[2:,:].flatten():
        ax.set_box_aspect(aspect)
    extent = (0, (P + 1) / P_year) * 2

    ex = CutOffExperiment(model, dps=[P_year], P=P, P_year=P_year)
    C_obs = ex.observed_covariance((N_samples,), L, seed=98765)

    for jBasis, Basis in enumerate((SmallStepBasis, TwoHopBasis)):
        basis = Basis(P)
        cclosures = basis.evaluate_covariance(C_obs, compl=True)
        for n in range(N_samples):
            triangle_plot(
                basis, cclosures[:, n], ax=axs[jBasis, n], show_xticklabels=False, extent=extent,
                show_yticklabels=(n == 0), vabs=triangleparms[0][0], cmap=triangleparms[0][1], cbar=False,
                aspect=aspect, ticks=xticks_year, ticklabels=xticks_year)
    for jax, ax in enumerate(axs[0,:]):
        ax.text(
            0.50, 1.07, f'sample {jax+1}', c='k', transform=ax.transAxes, ha='center',
            va='baseline')
    for ax in axs[-1,:]:
        ax.set_xticklabels(xticks_year)
        ax.text(0.5, -0.39, '$t$ [yr]', ha='center', va='baseline', transform=ax.transAxes)
    for jax, ax in enumerate(axs[:, 0]):
        ylabell, ylabelr = ylabels[jax]
        ax.text(-0.37, 0.50, ylabell, ha='right', va='center', transform=ax.transAxes, rotation=90)
        ax.text(-0.22, 0.50, ylabelr, ha='right', va='center', transform=ax.transAxes, rotation=90)
    cax_limits = [1.17, 0.1, 0.10, 0.80]
    for jax, ax in enumerate(axs[0:len(triangleparms), -1]):
        cax = ax.inset_axes(cax_limits)
        vabs = triangleparms[jax][0] / 180 * np.pi
        cbar = plt.colorbar(
            ScalarMappable(norm=Normalize(vmin=-vabs, vmax=vabs), cmap=triangleparms[jax][1]), cax,
            shrink=0.5, orientation='vertical', ticks=[-vabs, 0, vabs])
        cbar.set_ticklabels(triangleparms[jax][2])
    if fnout is None:
        plt.show()
    else:
        fig.savefig(fnout, dpi=450)

def speckle_bias_plot(fnout=None):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    P = 91
    P_year = 30
    N_samples = 1024
    L = 169
    xticks_year = (0, 1, 2, 3)
    model = model_catalog('decorrsoil', P_year=P_year, band='C')
    ylabels = [
        ('small step', '$\\tau$ [yr]'), ('two hop', '$ \\tau$ [yr]')]
    aspect = 0.75
    triangleparms = [
        (180, cmap_cyclic, ['$-\\pi$', '0', '$\\pi$']),
        (45, cmap_clipped, ['$-\\pi/4$', '0', '$\\pi/4$'])]
    fig, axs = prepare_figure(
        nrows=2, ncols=1, figsize=(0.85, 0.66), sharey='row', sharex='col', top=0.98, left=0.07,
        wspace=0.20, hspace=0.16, bottom=0.17, right=0.90)
    for ax in axs.flatten():
        ax.set_box_aspect(aspect)
    extent = (0, (P + 1) / P_year) * 2

    ex = CutOffExperiment(model, dps=[P_year], P=P, P_year=P_year)
    C_obs = ex.observed_covariance((N_samples,), L, seed=98765)

    for jBasis, Basis in enumerate((SmallStepBasis, TwoHopBasis)):
        basis = Basis(P)
        cclosures = basis.evaluate_covariance(C_obs, compl=True)
        cclosures_m = np.mean(cclosures, axis=1)
        triangle_plot(
            basis, cclosures_m, ax=axs[jBasis], show_xticklabels=False, extent=extent,
            show_yticklabels=True, vabs=triangleparms[0][0], cmap=triangleparms[0][1], cbar=False,
            aspect=aspect, ticks=xticks_year, ticklabels=xticks_year)
    ax = axs[-1]
    ax.set_xticklabels(xticks_year)
    ax.text(0.5, -0.35, '$t$ [yr]', ha='center', va='baseline', transform=ax.transAxes)
    for jax, ax in enumerate(axs):
        ylabell, ylabelr = ylabels[jax]
        ax.text(-0.37, 0.50, ylabell, ha='right', va='center', transform=ax.transAxes, rotation=90)
        ax.text(-0.22, 0.50, ylabelr, ha='right', va='center', transform=ax.transAxes, rotation=90)
    cax_limits = [1.17, 0.1, 0.10, 0.80]
    for jax, ax in enumerate(axs):
        cax = ax.inset_axes(cax_limits)
        vabs = triangleparms[jax][0] / 180 * np.pi
        cbar = plt.colorbar(
            ScalarMappable(norm=Normalize(vmin=-vabs, vmax=vabs), cmap=triangleparms[jax][1]), cax,
            shrink=0.5, orientation='vertical', ticks=[-vabs, 0, vabs])
        cbar.set_ticklabels(triangleparms[jax][2])
    if fnout is None:
        plt.show()
    else:
        fig.savefig(fnout, dpi=450)

if __name__ == '__main__':

    pathfig = Path('/home/simon/Work/closig/figures')
    fnout = pathfig / 'triangle_speckle.pdf'
    speckle_plot(fnout)
    
    fnout = pathfig / 'triangle_speckle_mean.pdf'
    # speckle_bias_plot(fnout)    
