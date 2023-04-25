'''
Created on Jan 19, 2023

@author: simon
'''
import numpy as np

def model_catalogue(scenario, Nyear=30):
    from model import DiffDispTile, VegModel
    if scenario == 'diffdisp':
        dphase = 2 * np.pi / Nyear
        intensities = [0.8, 0.2]
        model = DiffDispTile(intensities, [0.30, 0.30], [0.0, dphase], coh0=[0.6, 0.6])
    elif scenario == 'seasonalveg':
        h, theta = 0.4, 40 * np.pi / 180
        model = VegModel(
            wavelength=0.2, h=h, Fg=0.5, fc=0.5 / h, Nyear=Nyear, nrstd=0.01, nra=0.10, nrt=0.0, 
            theta=theta)
    elif scenario == 'seasonaltrendveg':
        h, theta = 0.4, 40 * np.pi / 180
        model = VegModel(
            wavelength=0.2, h=h, Fg=0.5, fc=0.5 / h, Nyear=Nyear, nrstd=0.00, nrt=0.10, nra=0.03,
            theta=theta)
    else:
        raise ValueError(f"Scenario {scenario} not known")
    return model

def double_plot(bases, models, fnout=None, ticks=None, ticklabels=None):
    from plotting import triangle_plot, prepare_figure, initialize_matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib import cm
    from matplotlib.colors import Normalize
    import colorcet as cc
    cmap, vabs = cc.cm['CET_C1'], 180
    initialize_matplotlib()
    fig = plt.figure(figsize=(6.5, 1.8))
    w = 5
    gsw = len(bases) * len(models) * w + (len(models) - 1)
    gs = gridspec.GridSpec(
        2, gsw, left=0.08, right=0.91, top=0.92, bottom=0.16, wspace=1.0, hspace=0.2)

    def create_sb(r, m):
        c0 = m * (w * len(bases) + 1)
        return [plt.subplot(gs[r, c0 + w * b:c0 + w * b + w]) for b in range(len(bases))]

    axs = [[create_sb(r, m) for m in range(len(models))] for r in range(2)]
    cclosures = {}
    jplot = 0
    for jm, nm in enumerate(models):
        _axs = axs[0][jm][:]
        x = 0.5 * (_axs[0].bbox.x0 + _axs[1].bbox.x1)
        x_fig = fig.transFigure.inverted().transform((x, 0))[0]
        fig.text(x_fig, 0.99, nm, c='k', ha='center', va='top', transform=fig.transFigure)
    for jm, nm in enumerate(models):
        model = models[nm]
        for jb, nb in enumerate(bases):
            basis = bases[nb]
            N = basis.N
            if len(cclosures) == 0:
                C = model.covariance(N, displacement_phase=False)
                cclosures['closures'] = basis.evaluate_covariance(C, compl=True)
                C0 = model.covariance(N, displacement_phase=True)
                dC = C * C0.conj() / np.abs(C0)
                cclosures['short_error'] = basis.evaluate_covariance(dC, compl=True, forward_only=True)
            triangle_plot(
                basis, cclosures['closures'], ax=axs[0][jm][jb], ticks=ticks, ticklabels=ticklabels,
                show_xticklabels=False, show_yticklabels=(jb == 0), blabel=nb, plabel=jplot, vabs=vabs, 
                cmap=cmap)
            axs[1][jm][jb].text(
                0.50, -0.47, '$t$ [years]', transform=axs[1][jm][jb].transAxes, ha='center', va='baseline')
            triangle_plot(
                basis, cclosures['short_error'], ax=axs[1][jm][jb], ticks=ticks, ticklabels=ticklabels,
                show_yticklabels=(jb == 0), plabel=jplot + 1, cmap=cmap, vabs=vabs)
            jplot = jplot + 2
            cclosures = {}

    cax = fig.add_axes((0.928, 0.300, 0.015, 0.400))
    # cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar = fig.colorbar(cm.ScalarMappable(norm=Normalize(-vabs, vabs, clip=True), cmap=cmap), cax=cax)
    cbar.set_ticks([-180, -90, 0, 90, 180])
    cbarlabel = 'phase [$^{\\circ}$]'
    cax.text(2.4, 1.2, cbarlabel, ha='center', va='baseline', transform=cax.transAxes)

    rowlabels = ['closure [rad]', 'forw. error [rad]']
    for jax, ax in enumerate([_ax[0][0] for _ax in axs]):
        ax.text(
            -0.25, 0.50, '$\\tau$ [years]', rotation=90, transform=ax.transAxes, ha='right', va='center')
        ax.text(
            -0.50, 0.50, rowlabels[jax], rotation=90, transform=ax.transAxes, ha='right', va='center',
            c='k')
    if fnout is not None:
        plt.savefig(fnout)

if __name__ == '__main__':
    from expansion import TwoHopBasis, SmallStepBasis
    N = 91
    Nyear = 30
    ticks = np.array([0, 30, 60, 90]) + 1
    ticklabels = [0, 1, 2, 3]
    bases = {'t': TwoHopBasis(N), 's': SmallStepBasis(N)}
    scennames = ['diffdisp', 'seasonalveg', 'seasonaltrendveg']
    m = [model_catalogue(scenname, Nyear=Nyear) for scenname in scennames]
    models = {'differential subsidence': m[0], 'seasonal vegetation': m[1], 'seasonal vegetation + trend': m[2]}
    fnout = '/home/simon/Work/closig/figures/model.pdf'
    double_plot(bases, models, fnout=fnout, ticks=ticks, ticklabels=ticklabels)
    
    # to do: show sensitivity to single phase difference
    # spatial plots of tau=1 year closure phases
    # try normalization based on intensity(geom. mean)
    