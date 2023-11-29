from matplotlib import pyplot as plt
import numpy as np

from closig.experiments import (
    CutOffExperiment, model_catalog, phase_error_plot, phase_history_bias_plot, phase_history_metric_plot,
    triangle_experiment_plot)
from closig.visualization import prepare_figure, cmap_clipped, cmap_cyclic, colslist
from closig import TwoHopBasis, SmallStepBasis

P = 91
P_year = 30
dps = [1, 15, 30, 90]
xlim_year = (0, P / P_year)
xticks_year = (0, 1, 2, 3)

modelp = model_catalog('precipsoil', P_year=P_year, band='C')
modelss = model_catalog('seasonalsoil', P_year=P_year, band='C')
modelps = model_catalog('seasonalprecipsoil', P_year=P_year, band='C')
models = model_catalog('seasonalveg', P_year=P_year, band='C')
modelt = model_catalog('seasonaltrendveg', P_year=P_year, band='C')
modeld = model_catalog('diffdisp', P_year=P_year, band='C')

models = [modelp, modelss, modelps, models, modelt, modeld]
headers = [
    'event SM', 'seasonal SM', 'harmonic SM', 'seasonal veg.',
    'growing veg.', 'differential motion']
ylabels = [
    ('small step', '$\\tau$ [yr]'), ('two hop', '$\\tau$ [yr]'), ('$\\varphi$ bias', '[rad]'),
    ('$\\phi$ bias', '[rad]'), ('$c_{\\phi}$', '[-]')]
aspect = 0.75
triangleparms = [
    (180, cmap_cyclic, ['$-\\pi$', '0', '$\\pi$']),
    (45, cmap_clipped, ['$-\\pi/4$', '0', '$\\pi/4$'])]
dps_labels = ['nearest', '1/2 yr', '1 yr', '3 yr']
from matplotlib.cm import ScalarMappable
import matplotlib.lines as mlines
from matplotlib.colors import Normalize
fig, axs = prepare_figure(
    nrows=5, ncols=len(models), figsize=(2.04, 1.35), sharey='row', sharex='col', top=0.97, left=0.06,
    wspace=0.20, hspace=0.18, bottom=0.07, right=0.91)
for ax in axs[2:,:].flatten():
    ax.set_box_aspect(aspect)

for jmodel, model in enumerate(models):
    ex = CutOffExperiment(model, dps=dps, P=P, P_year=P_year)
    show_ylabel = False
    extent = xlim_year * 2
    triangle_experiment_plot(
        ex, Basis=SmallStepBasis, ax=axs[0, jmodel], aspect=aspect, show_xticklabels=False,
        ticks=xticks_year, vabs=triangleparms[0][0], cmap=triangleparms[0][1])
    triangle_experiment_plot(
        ex, Basis=TwoHopBasis, ax=axs[1, jmodel], aspect=aspect, show_xticklabels=False,
        ticks=xticks_year, vabs=triangleparms[1][0], cmap=triangleparms[1][1])
    phase_error_plot(
        ex, ax=axs[2, jmodel], show_ylabel=show_ylabel, show_xlabel=False, show_xticklabels=False)
    phase_history_bias_plot(
        ex, ax=axs[3, jmodel], show_ylabel=show_ylabel, show_xlabel=False, show_xticklabels=False)
    phase_history_metric_plot(ex, ax=axs[4, jmodel], show_ylabel=show_ylabel, show_xticklabels=False)
    for ax in axs[-3:, jmodel]:
        ax.set_xticks(xticks_year)
        ax.set_xlim(xlim_year)
        ax.grid(True, axis='x', color='#666666', lw=0.5, alpha=0.1)
    axs[-1, jmodel].set_xticklabels(xticks_year)
    axs[0, jmodel].text(
        0.50, 1.06, headers[jmodel], c='k', transform=axs[0, jmodel].transAxes, ha='center', va='baseline')
for jax, ax in enumerate(axs[:, 0]):
    ylabell, ylabelr = ylabels[jax]
    ax.text(-0.37, 0.50, ylabell, ha='right', va='center', transform=ax.transAxes, rotation=90)
    ax.text(-0.22, 0.50, ylabelr, ha='right', va='center', transform=ax.transAxes, rotation=90)
cax_limits = [1.17, 0.1, 0.10, 0.80]
for jax, ax in enumerate(axs[0:len(triangleparms), -1]):
    cax = ax.inset_axes(cax_limits)
    vabs = triangleparms[jax][0] / 180 * np.pi
    cbar = plt.colorbar(
        ScalarMappable(norm=Normalize(vmin=-vabs, vmax=vabs), cmap=triangleparms[jax][1]), cax, shrink=0.5,
        orientation='vertical', ticks=[-vabs, 0, vabs])
    cbar.set_ticklabels(triangleparms[jax][2])
ax = axs[-2, -1]
handles = [mlines.Line2D([], [], color=c, label=l, lw=0.8) for l, c in zip(dps_labels, colslist)]
ax.legend(
    handles=handles, bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.0, frameon=False,
    borderpad=0.3, handlelength=1.0, handletextpad=0.6)
plt.show()

