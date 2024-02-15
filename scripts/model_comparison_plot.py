from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.lines as mlines
from matplotlib.colors import Normalize
import numpy as np

from closig.experiments import (
    CutOffExperiment, model_catalog, phase_error_plot, phase_history_bias_plot, phase_history_metric_plot,
    triangle_experiment_plot, phase_history_difference_plot)
from closig.visualization import prepare_figure, cmap_clipped, cmap_cyclic, colslist
from closig import TwoHopBasis, SmallStepBasis

def plot_models(models, P, P_year=30, dps=[1, 30], lamb=0.055, fnout=None):
    xlim_year = (0, P / P_year)
    xticks_year = range(P // P_year + 1)  # (0, 1, 2, 3)

    conv = (lambda phi: phi * lamb / (4 * np.pi), lambda d: 4 * np.pi * d / lamb)

    headers = [
        'event SM', 'harmonic SM', 'seasonal SM', 'seasonal veg.',
        'growing veg.', 'differential motion']
    ylabels = [
        ('small step', '$\\tau$ [yr]'), ('two hop', '$\\tau$ [yr]'), ('$\\varphi$ bias', '[rad]'),
        ('$\\phi$ bias', '[rad]'), ('$c_{\\phi}$', '[-]')]
    aspect = 0.75
    triangleparms = [
        (180, cmap_cyclic, ['$-\\pi$', '0', '$\\pi$']),
        (45, cmap_clipped, ['$-\\pi/4$', '0', '$\\pi/4$'])]
    dps_labels = ['nearest', '1/2 yr', '3 yr']  # ['nearest', '1/2 yr', '1 yr', '3 yr']
    alphal = 0.8

    fig, axs = prepare_figure(
        nrows=5, ncols=len(models), figsize=(2.04, 1.31), sharey='row', sharex='col', top=0.975, 
        left=0.061, wspace=0.200, hspace=0.140, bottom=0.070, right=0.910)
    for ax in axs[2:,:].flatten():
        ax.set_box_aspect(aspect)

    for jmodel, model in enumerate(models):
        ex = CutOffExperiment(model, dps=dps, P=P, P_year=P_year)
        show_ylabel = False
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
            ex, ax=axs[3, jmodel], alpha=alphal, show_ylabel=show_ylabel, show_xlabel=False,
            show_xticklabels=False)
        # phase_history_difference_plot(
        #     ex, ax=axs[3, jmodel], show_ylabel=show_ylabel, show_xlabel=False, show_xticklabels=False)
        phase_history_metric_plot(
            ex, ax=axs[4, jmodel], alpha=alphal, show_ylabel=show_ylabel, show_xticklabels=False, 
            y_xlab=-0.51)
        for ax in axs[-3:, jmodel]:
            ax.set_xticks(xticks_year)
            ax.set_xlim(xlim_year)
            ax.grid(True, axis='x', color='#666666', lw=0.5, alpha=0.1)
        axs[-1, jmodel].set_xticklabels(xticks_year)
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
    for jax, ax in enumerate(axs[len(triangleparms):len(triangleparms)+2, -1]):
        secax = ax.secondary_yaxis('right', functions=conv)
        secax.set_yticks((-1e-2, 0, 1e-2))
        secax.set_yticklabels((-1, 0, 1))
        ax.text(1.24, 0.50, '[cm]', rotation=270, ha='left', va='center', transform=ax.transAxes)
    ax = axs[-1, -1]
    handles = [mlines.Line2D([], [], color=c, label=l, lw=0.8, alpha=alphal)
               for l, c in zip(dps_labels, colslist)]
    ax.legend(
        handles=handles, bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.0, frameon=False,
        borderpad=0.3, handlelength=1.0, handletextpad=0.6)
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
    modelsv = model_catalog('seasonalveg', P_year=P_year, band='C')
    modelt = model_catalog('seasonaltrendveg', P_year=P_year, band='C')
    modeld = model_catalog('diffdisp', P_year=P_year, band='C')

    models = [modelp, modelhs, modelps, modelsv, modelt, modeld]
    # models = [modelp, modelhs]
    # seasonal veg model demonstrate that consistent long-term interferograms are insufficient [as they are all wrong; substantial closure errors at tau approx 1/2 yr]

    fnout = '/home/simon/Work/closig/figures/model.pdf'
    # phi bias and ph sign convention: dphi[k] = phi[k] - phi[0]
    plot_models(models, P, P_year=P_year, dps=dps, fnout=fnout)

    print(np.angle(modelp._coherence_element(4, 5)))
