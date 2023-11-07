from matplotlib import pyplot as plt
import numpy as np

from experiments import (
    CutOffExperiment, model_catalog, phase_error_plot, phase_history_bias_plot, phase_history_metric_plot)
from closig.visualization import prepare_figure


P = 91
P_year = 30
dps = [1, 15, 30, 90]

modelp = model_catalog('precipsoil', P_year=P_year, band='C')
models = model_catalog('seasonalveg', P_year=P_year, band='C')
modeld = model_catalog('diffdisp', P_year=P_year, band='C')


models = [modelp, models, modeld]
# models = [modelp]
fig, axs = prepare_figure(
    nrows=5, ncols=len(models), figsize=(1.5, 1.35), sharey='row', sharex=False, bottom=0.12, left=0.09, 
    wspace=0.2)


for jmodel, model in enumerate(models):
    ex = CutOffExperiment(model, dps=dps, P=P, P_year=P_year)
    show_ylabel = (jmodel == 0)
    # add both triangle plots
    phase_error_plot(ex, ax=axs[2, jmodel], show_ylabel=show_ylabel, show_xlabel=False)
    phase_history_bias_plot(ex, ax=axs[3, jmodel], show_ylabel=show_ylabel, show_xlabel=False)
    phase_history_metric_plot(ex, ax=axs[4, jmodel], show_ylabel=show_ylabel)
    # add legends
    for ax in axs[-3:, jmodel]:
        ax.set_xticks([0, 1, 2, 3])
plt.show()

