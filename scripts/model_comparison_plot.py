from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.lines as mlines
from matplotlib.colors import Normalize
import numpy as np

from closig.experiments import (
    CutOffExperiment, model_catalog, phase_error_plot, phase_history_bias_plot, phase_history_metric_plot,
    triangle_experiment_plot, phase_history_difference_plot, TrendSeasonalMetric)
from closig.visualization import prepare_figure, cmap_clipped, cmap_cyclic, colslist
from closig import TwoHopBasis, SmallStepBasis

def plot_models(models, P, P_year=30, dps=[1, 30], lamb=0.055, headers=None, fnout=None):
    xlim_year = (0, P / P_year)
    xticks_year = range(P // P_year + 1)  # (0, 1, 2, 3)

    conv = (lambda phi: phi * lamb / (4 * np.pi), lambda d: 4 * np.pi * d / lamb)

    ylabels = [
        ('small step', '$\\tau$ [yr]'), ('two hop', '$\\tau$ [yr]'), ('$\\Phi_{\\cdot1}$ bias', '[rad]'),
        ('$\\theta$ bias', '[rad]')]
    aspect = 0.75
    triangleparms = [
        (180, cmap_cyclic, ['$-\\pi$', '0', '$\\pi$']),
        (45, cmap_clipped, ['$-\\pi/4$', '0', '$\\pi/4$'])]
    dps_labels = ['nearest', '1/2 yr', 'all']  # ['nearest', '1/2 yr', '1 yr', 'all]
    alphal = 0.8

    metric = TrendSeasonalMetric(P_year=P_year)

    fig, axs = prepare_figure(
        nrows=4, ncols=len(models), figsize=(2.00, 1.12), sharey='row', sharex='col', top=0.96,
        left=0.071, wspace=0.200, hspace=0.150, bottom=0.091, right=0.901)
    for ax in axs[len(triangleparms):,:].flatten():
        ax.set_box_aspect(aspect)

    for jmodel, model in enumerate(models):
        ex = CutOffExperiment(model, dps=dps, P=P, P_year=P_year)
        show_ylabel = False
        ph = ex.phase_history()
        print('trendmetric', metric.evaluate(ph[..., 0,:], ph[..., -1,:]))
        triangle_experiment_plot(
            ex, Basis=SmallStepBasis, ax=axs[0, jmodel], aspect=aspect, show_xticklabels=False,
            ticks=xticks_year, vabs=triangleparms[0][0], cmap=triangleparms[0][1])
        triangle_experiment_plot(
            ex, Basis=TwoHopBasis, ax=axs[1, jmodel], aspect=aspect, show_xticklabels=False,
            ticks=xticks_year, vabs=triangleparms[1][0], cmap=triangleparms[1][1])
        phase_error_plot(
            ex, ax=axs[2, jmodel], show_ylabel=show_ylabel, show_xlabel=False, show_xticklabels=False,
            phase_history_sign=True)
        phase_history_bias_plot(
            ex, ax=axs[3, jmodel], alpha=alphal, show_ylabel=show_ylabel, y_xlab=-0.43,
            show_xticklabels=False)
        # phase_history_difference_plot(
        #     ex, ax=axs[3, jmodel], show_ylabel=show_ylabel, show_xlabel=False, show_xticklabels=False)
        # phase_history_metric_plot(
        #     ex, ax=axs[4, jmodel], alpha=alphal, show_ylabel=show_ylabel, show_xticklabels=False,
        #     samples=samples, y_xlab=-0.43)
        for ax in axs[len(triangleparms):, jmodel]:
            ax.set_xticks(xticks_year)
            ax.set_xlim(xlim_year)
            ax.grid(True, axis='x', color='#666666', lw=0.5, alpha=0.1)
        axs[-1, jmodel].set_xticklabels(xticks_year)
        if headers is not None:
            axs[0, jmodel].text(
                0.500, 1.075, headers[jmodel], c='k', transform=axs[0, jmodel].transAxes, ha='center',
                va='baseline')
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
    for jax, ax in enumerate(axs[len(triangleparms):len(triangleparms) + 2, -1]):
        secax = ax.secondary_yaxis('right', functions=conv)
        secax.set_yticks((-1e-2, 0, 1e-2))
        yticklabels = (None, 0, 1) if jax==1 else (-1, 0, 1)
        secax.set_yticklabels(yticklabels)
        ax.text(1.24, 0.50, '[cm]', rotation=270, ha='left', va='center', transform=ax.transAxes)
    ax = axs[-1, -1]
    handles = [mlines.Line2D([], [], color=c, label=l, lw=0.8, alpha=alphal)
               for l, c in zip(dps_labels, colslist)]
    ax.legend(
        handles=handles, bbox_to_anchor=(1.10, -0.12), loc='center left', borderaxespad=0.0, frameon=False,
        borderpad=0.3, handlelength=0.6, handletextpad=0.4)
    if fnout is None:
        plt.show()
    else:
        fig.savefig(fnout, dpi=450)

if __name__ == '__main__':
    P = 91
    P_year = 30
    dps = [1, 15, 90]  # [1, 15, 30, 90]

    modelp = model_catalog('precipsoil', P_year=P_year, band='C')
    modelss = model_catalog('seasonalsoil', P_year=P_year, band='C')
    modelps = model_catalog('seasonalprecipsoil', P_year=P_year, band='C')
    modelhs = model_catalog('harmonicsoil', P_year=P_year, band='C')
    modelv = model_catalog('veg', P_year=P_year, band='C')
    modelsv = model_catalog('seasonalveg', P_year=P_year, band='C')
    modelsvd = model_catalog('seasonalvegd', P_year=P_year, band='C')
    modeltv = model_catalog('seasonaltrendveg', P_year=P_year, band='C')
    modeltvd = model_catalog('seasonaltrendvegd', P_year=P_year, band='C')
    modeld = model_catalog('diffdisp', P_year=P_year, band='C')
    modeldd = model_catalog('diffdispd', P_year=P_year, band='C')

    models = [modelp, modelhs, modelps, modelsv, modeltv, modeld]


    # seasonal veg model demonstrate that consistent long-term interferograms are insufficient [as they are all wrong; substantial closure errors at tau approx 1/2 yr]

    headers = ['S1 event SM', 'S2 harmonic SM', 'S3 seasonal SM', 'S4 diff. motion', 
               'S5 decorrelating diff. motion']
    models = [modelp, modelhs, modelps, modeld, modeldd]    
    fnout = '/home/simon/Work/closig/figures/model.pdf'
    plot_models(models, P, P_year=P_year, dps=dps, fnout=fnout, headers=headers)

    headers = [
        'S6 stationary veg.', 'S7 seasonal veg.', 'S8 decorr. seasonal veg.', 'S9 drying veg.', 
        'S10 decorr. drying veg.']
    models = [modelv, modelsv, modelsvd, modeltv, modeltvd]    
    fnout = '/home/simon/Work/closig/figures/modelv.pdf'
    plot_models(models, P, P_year=P_year, dps=dps, fnout=fnout, headers=headers)

    # phi bias and ph sign convention: dphi[k] = phi[k] - phi[0]
    # print(np.angle(modelp.covariance(P)[0, :7]))
