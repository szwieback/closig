'''
    Simulating a semi-arid environment with sparse shrubbery with seasonal moisture content
'''
from closig.model import SeasonalVegLayer, HomogSoilLayer, LayeredCovModel, TiledCovModel, ContHetDispModel, ScattSoilLayer
from closig.expansion import SmallStepBasis, TwoHopBasis
from scripts.plotting import triangle_plot
from matplotlib import pyplot as plt
from closig.linking import EMI, EMI_py, CutOffRegularizer, NearestNeighbor, EVD
import numpy as np

# Two shrub layers with different moisture content. The first is a dummy with 0 height to simulate bare soil
shrubs_a = SeasonalVegLayer(n_mean=1.5 - 0.01j, n_std=0, n_amp=1,
                            n_t=0, P_year=30, density=0, dcoh=1, h=0, name='Bare Soil')

shrubs_b = SeasonalVegLayer(n_mean=1.2 - 0.01j, n_std=0, n_amp=0.2,
                            n_t=0, P_year=30, density=0.1, dcoh=1, h=0.1, name='Shrubs')

shrubs_b = SeasonalVegLayer(n_mean=1.2 - 0.01j, n_std=0, n_amp=0,
                            n_t=0.1, P_year=30, density=0.05, dcoh=1, h=0.2, name='Shrubs')


# Tile shrub layers together
shrubs = TiledCovModel([shrubs_a, shrubs_b], fractions=[
                       0, 1], name='Shrubs')
# Static soil layer
# soil = HomogSoilLayer(name='Soil')

# Layered Model
# model = LayeredCovModel([shrubs, soil])
model = TiledCovModel([shrubs_a, shrubs_b], fractions=[
    0.6, 0.4], name='Shrubs')

# Analysis & Plotting
P = 60

fig, ax = plt.subplots(nrows=1, ncols=3)
model.plot_matrices(
    P, coherence=True, displacement_phase=False, ax=[ax[0], ax[1]])

# Compute closure phases and plot
smallStep = SmallStepBasis(P)
closure_phases = smallStep.evaluate_covariance(
    shrubs.covariance(P), compl=True, normalize=False)

print(f'Mean Closure Phase: {np.angle(closure_phases).mean()}')
triangle_plot(smallStep, closure_phases,
              ax=ax[2], cmap=plt.cm.seismic, vabs=45)
ax[2].set_title(
    f'Small Step Basis\n mean: {np.round(np.angle(closure_phases).mean(), 3)} rad')
plt.show()

# plt.imshow(model.get_baselines(P))
# plt.show()

plt.scatter(model.get_baselines(P).flatten(),
            np.angle(shrubs.covariance(P)).flatten())
plt.show()

taus = [1, 15, 20, 30, 45, 60]

for tau in taus:
    G = np.abs(shrubs.covariance(P, coherence=True))
    G = CutOffRegularizer().regularize(G, tau_max=tau)
    cov = shrubs.covariance(P, coherence=True)
    cov = G * np.exp(1j * np.angle(cov))

    # np.linalg.cholesky(cov)
    pl_evd = EVD().link(cov, G=G, corr=True)
    plt.plot(np.angle(pl_evd), label=f'tau: {tau}')

plt.legend(loc='best')
plt.grid()
plt.show()

# plt.imshow(np.abs(G), cmap=plt.cm.gray)
# plt.show()
# Link
