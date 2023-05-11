'''
    This script is for exploratory forward model configurations, probably redundant
    
'''
from closig.model import SeasonalVegLayer, HomogSoilLayer, LayeredCovModel, TiledCovModel, ContHetDispModel
from closig.expansion import SmallStepBasis, TwoHopBasis
from plotting import triangle_plot
from matplotlib import pyplot as plt
import numpy as np

shrubs = SeasonalVegLayer(n_mean=1.2 - 0.01j, n_std=0.0001, n_amp=0.1,
                          n_t=0, P_year=30, density=0.01, dcoh=0.9, h=0.1, name='Shrubs')

canopy = SeasonalVegLayer(n_mean=1.2 - 0.01j, n_std=0.0001, n_amp=0.1,
                          n_t=0.5, P_year=30, density=0.01, dcoh=0.5, h=0.05, name='Canopy')

shrubs_b = SeasonalVegLayer(n_mean=1.2 - 0.01j, n_std=0.0001, n_amp=0.5,
                            n_t=0, P_year=30, density=0.01, dcoh=0.9, h=0.1, name='Shrubs')


center = HomogSoilLayer(dz=-0.01, dcoh=1, name='Center')
trough = HomogSoilLayer(dz=-0.02, dcoh=1, name='Trough')
polygons = ContHetDispModel(
    means=[-0.25, -0.05], stds=[0.01, 0.01], L=1000, weights=[0.15, 0.85], name='Continuous Polygons')

disp = TiledCovModel([center, trough], fractions=[
    0.8, 0.2], name='Heterogenous Displacement')
P = 10

fig, ax = plt.subplots(nrows=1, ncols=3)

polygons.plot_matrices(
    P, coherence=True, displacement_phase=False, ax=[ax[0], ax[1]])

# Compute closure phases and plot
smallStep = SmallStepBasis(P)
closure_phases = smallStep.evaluate_covariance(
    polygons.covariance(P), compl=True, normalize=False)

print(f'Mean Closure Phase: {np.angle(closure_phases).mean()}')
triangle_plot(smallStep, closure_phases,
              ax=ax[2], cmap=plt.cm.seismic, vabs=180)
ax[2].set_title(
    f'Small Step Basis\n mean: {np.round(np.angle(closure_phases).mean(), 3)} rad')

plt.show()
