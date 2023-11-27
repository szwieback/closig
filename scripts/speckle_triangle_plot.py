from matplotlib import pyplot as plt
import numpy as np

from closig.experiments import (
    CutOffExperiment, model_catalog, phase_error_plot, phase_history_bias_plot, phase_history_metric_plot,
    triangle_experiment_plot)
from closig.visualization import prepare_figure, cmap_clipped, cmap_cyclic, colslist, triangle_plot
from closig import TwoHopBasis, SmallStepBasis

P = 91
P_year = 30
N_samples = 5
L = 169

xlim_year = (0, P / P_year)
xticks_year = (0, 1, 2, 3)

model = model_catalog('decorrsoil', P_year=P_year, band='C')
ylabels = [
    ('small step', '[yr]'), ('two hop', '[yr]')]
aspect = 0.75
triangleparms = [
    (180, cmap_cyclic, ['$-\\pi$', '0', '$\\pi$']),
    (45, cmap_clipped, ['$-\\pi/4$', '0', '$\\pi/4$'])]
dps_labels = ['nearest', '1/2 yr', '1 yr', '3 yr']
from matplotlib.cm import ScalarMappable
import matplotlib.lines as mlines
from matplotlib.colors import Normalize
fig, axs = prepare_figure(
    nrows=2, ncols=N_samples, figsize=(2.04, 0.63), sharey='row', sharex='col', top=0.94, left=0.07,
    wspace=0.20, hspace=0.16, bottom=0.14, right=0.90)
for ax in axs[2:,:].flatten():
    ax.set_box_aspect(aspect)
extent = (0, (P + 1) / P_year) * 2

ex = CutOffExperiment(model, dps=[P_year], P=P, P_year=P_year)
C_obs = ex.observed_covariance((N_samples,), L, seed=1111)

for jBasis, Basis in enumerate((SmallStepBasis, TwoHopBasis)):
    basis = Basis(P)
    cclosures = basis.evaluate_covariance(C_obs, compl=True)
    for n in range(N_samples):
        triangle_plot(
            basis, cclosures[:, n], ax=axs[jBasis, n], show_xticklabels=False, show_yticklabels=False,
            extent=extent, vabs=triangleparms[0][0], cmap=triangleparms[0][1], cbar=False,
            aspect=aspect, ticks=xticks_year)
for jax, ax in enumerate(axs[0, :]):
    ax.text(
        0.50, 1.06, f'sample {jax+1}', c='k', transform=ax.transAxes, ha='center', 
        va='baseline')
for ax in axs[-1, :]:
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
        ScalarMappable(norm=Normalize(vmin=-vabs, vmax=vabs), cmap=triangleparms[jax][1]), cax, shrink=0.5,
        orientation='vertical', ticks=[-vabs, 0, vabs])
    cbar.set_ticklabels(triangleparms[jax][2])
plt.show()

