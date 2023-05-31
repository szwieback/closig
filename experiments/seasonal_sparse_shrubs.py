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
shrubs_a = SeasonalVegLayer(n_mean=1.5 - 0.01j, n_std=0, n_amp=0,
                            n_t=0, P_year=30, density=0, dcoh=1, h=0, name='Bare Soil')

shrubs_b = SeasonalVegLayer(n_mean=1.2 - 0.01j, n_std=0, n_amp=0.01,
                            n_t=0.0, P_year=30, density=1, dcoh=1, h=1, name='Shrubs')

# shrubs_b = SeasonalVegLayer(n_mean=1.2 - 0.01j, n_std=0, n_amp=0,
#                             n_t=0.1, P_year=30, density=0.05, dcoh=1, h=0.2, name='Shrubs')


# Tile shrub layers together

model = TiledCovModel([shrubs_a, shrubs_b], fractions=[
    0, 1], name='Shrubs')

# Analysis & Plotting

# Analysis & Plotting
P = 45
# model.plot_s()
# model.plot_mv(P)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
model.plot_matrices(
    P, coherence=True, displacement_phase=False, ax=[ax[0, 0], ax[0, 1]])

# Compute closure phases and plot
smallStep = SmallStepBasis(P)
twoHop = TwoHopBasis(P)
c_phases_ss = smallStep.evaluate_covariance(
    model.covariance(P), compl=True, normalize=False)

c_phases_th = twoHop.evaluate_covariance(
    model.covariance(P), compl=True, normalize=False)
print(f'Mean Closure Phase: {np.angle(c_phases_ss).mean()}')

triangle_plot(smallStep, c_phases_ss,
              ax=ax[1, 1], cmap=plt.cm.seismic, vabs=180)
triangle_plot(twoHop, c_phases_th,
              ax=ax[1, 0], cmap=plt.cm.seismic, vabs=45)
ax[1, 1].set_title(
    f'Small Step Basis\n mean: {np.round(np.angle(c_phases_ss).mean(), 3)} rad')
ax[1, 0].set_title(
    f'Two Hop Basis\n mean: {np.round(np.angle(c_phases_th).mean(), 3)} rad')
plt.tight_layout()
plt.show()


taus = np.arange(1, P, 10)
l2error = np.zeros((len(taus)))
colors = plt.cm.viridis(np.linspace(0, 1, len(taus)))

for tau, color, i in zip(taus, colors, range(len(taus))):
    G = np.abs(model.covariance(P, coherence=True))
    G = CutOffRegularizer().regularize(G, tau_max=tau)
    cov = model.covariance(P, coherence=True)
    cov = cov * G

    # np.linalg.cholesky(cov)
    pl_evd = EVD().link(cov, G=G)
    l2error[i] = np.linalg.norm(np.angle(pl_evd), 2)
    plt.plot(np.angle(pl_evd), label=f'tau: {tau}', color=color)

plt.legend(loc='best')
plt.xlabel('t')
plt.ylabel('Phase [rad]')
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

plt.plot(taus, l2error)
plt.xlabel('bw')
plt.ylabel('L2 error')
plt.show()

'''
    The tiling of the shrub layer has minimal affect on the phase linking.
    This experiment is characteristic standard seasonal vegetation.

    Error is largest in interferograms with temporal baselines of quarter of the period.
    Smallest errors are in the interferograms with baselines 0, 30, and 60


'''
