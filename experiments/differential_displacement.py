from matplotlib import pyplot as plt
import numpy as np

from experiments import (
    CutOffExperiment, model_catalog, phase_error_plot, phase_history_bias_plot, phase_history_metric_plot,
    triangle_experiment_plot)
from closig.visualization import prepare_figure, cmap_clipped, cmap_cyclic
from closig import TwoHopBasis, SmallStepBasis


P = 91
P_year = 30
dps = [1, 15, 30, 90]
xlim_year = (0, P/P_year)
xticks_year = (0, 1, 2, 3)

modelp = model_catalog('precipsoil', P_year=P_year, band='C')
models = model_catalog('seasonalveg', P_year=P_year, band='C')
modeld = model_catalog('diffdisp', P_year=P_year, band='C')


models = [modelp, models, modeld, modeld]

fig, axs = prepare_figure(
    nrows=5, ncols=len(models), figsize=(2.00, 1.45), sharey='row', sharex=False, top=0.98, left=0.09, 
    wspace=0.20, hspace=0.18, bottom=0.07, right=0.88)
aspect = 0.75
for ax in axs[2:, :].flatten():
    ax.set_box_aspect(aspect)


for jmodel, model in enumerate(models[:1]):
    ex = CutOffExperiment(model, dps=dps, P=P, P_year=P_year)
    show_ylabel = (jmodel == 0)
    extent = xlim_year * 2
    triangle_experiment_plot(
        ex, Basis=SmallStepBasis, ax=axs[0, jmodel], aspect=aspect, show_xticklabels=False, ticks=xticks_year, vabs=180, cmap=cmap_cyclic)
    triangle_experiment_plot(
        ex, Basis=TwoHopBasis, ax=axs[1, jmodel], aspect=aspect, show_xticklabels=False, ticks=xticks_year, vabs=90, cmap=cmap_clipped)    
    phase_error_plot(ex, ax=axs[2, jmodel], show_ylabel=show_ylabel, show_xlabel=False, show_xticklabels=False)
    phase_history_bias_plot(ex, ax=axs[3, jmodel], show_ylabel=show_ylabel, show_xlabel=False, show_xticklabels=False)
    phase_history_metric_plot(ex, ax=axs[4, jmodel], show_ylabel=show_ylabel)
    # add legends/cbars
    # add headers
    # fix labels
    for ax in axs[-3:, jmodel]:
        ax.set_xticks(xticks_year)
        ax.set_xlim(xlim_year)
plt.show()

