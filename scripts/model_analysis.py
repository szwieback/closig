'''
Created on Jan 19, 2023

@author: simon
'''

def double_plot(bases, models, fnout=None, ticks=None, ticklabels=None):
    from closig.visualization import triangle_plot, prepare_figure, initialize_matplotlib, cmap_cyclic
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib import cm
    from matplotlib.colors import Normalize
    cmap, vabs = cmap_cyclic, 180
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
            P = basis.P
            if len(cclosures) == 0:
                C = model.covariance(P, displacement_phase=False)
                cclosures['closures'] = basis.evaluate_covariance(C, compl=True)
                C0 = model.covariance(P, displacement_phase=True)
                dC = C * C0.conj() / np.abs(C0)
                cclosures['short_error'] = basis.evaluate_covariance(dC, compl=True, forward_only=True)
            triangle_plot(
                basis, cclosures['closures'], ax=axs[0][jm][jb], ticks=ticks, ticklabels=ticklabels,
                show_xticklabels=False, show_yticklabels=(jb == 0), blabel=nb, plabel=jplot, vabs=vabs,
                cmap=cmap, cbar=False)
            axs[1][jm][jb].text(
                0.50, -0.47, '$t$ [years]', transform=axs[1][jm][jb].transAxes, ha='center', va='baseline')
            triangle_plot(
                basis, cclosures['short_error'], ax=axs[1][jm][jb], ticks=ticks, ticklabels=ticklabels,
                show_yticklabels=(jb == 0), plabel=jplot + 1, cmap=cmap, vabs=vabs, cbar=False)
            jplot = jplot + 2
            cclosures = {}

    cax = fig.add_axes((0.928, 0.300, 0.015, 0.400))
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
        fig.savefig(fnout, dpi=450)
    else:
        plt.show()

if __name__ == '__main__':
    from closig import TwoHopBasis, SmallStepBasis
    from experiments import model_catalog
    import numpy as np
    P = 91
    P_year = 30
    ticks = np.array([0, 30, 60, 90]) + 1
    ticklabels = [0, 1, 2, 3]
    bases = {'t': TwoHopBasis(P), 's': SmallStepBasis(P)}
    scennames = ['diffdisp', 'seasonalveg', 'seasonaltrendveg', 'precipsoil']
    m = [model_catalog(scenname, P_year=P_year) for scenname in scennames]
    models = {
        'differential subsidence': m[0], 'seasonal vegetation': m[1], 'seasonal vegetation + trend': m[2]}
    models = {
        'differential subsidence': m[0], 'soil': m[3], 'seasonal vegetation': m[1]}    
    fnout = '/home/simon/Work/closig/figures/model.pdf'
    double_plot(bases, models, fnout=fnout, ticks=ticks, ticklabels=ticklabels)

