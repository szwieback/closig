'''
    How does velocity bias depend on precipitation frequency?
'''
from closig.model import PrecipScatterSoilLayer
from closig.expansion import SmallStepBasis, TwoHopBasis
from scripts.plotting import triangle_plot
from matplotlib import pyplot as plt
from closig.linking import EMI, EMI_py, CutOffRegularizer, NearestNeighbor, EVD
import numpy as np


# Tile shrub layers together

model = PrecipScatterSoilLayer(
    f=10, tau=1, dcoh=0.99, coh0=0, offset=0.1, scale=0.05)


# Analysis & Plotting
P = 90
model.plot_n(P)

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

plt.scatter(model.get_baselines(P).flatten(),
            np.angle(model.covariance(P)).flatten())
plt.xlabel('Baseline [t]')
plt.ylabel('Phase [rad]')
plt.show()

taus = [1, 2, 3, 5, 15, 45, 60]

for tau in taus:
    G = np.abs(model.covariance(P, coherence=True))
    G = CutOffRegularizer().regularize(G, tau_max=tau)
    cov = model.covariance(P, coherence=True)
    cov = G * np.exp(1j * np.angle(cov))

    # np.linalg.cholesky(cov)
    pl_evd = EVD().link(cov, G=G, corr=True)
    plt.plot(np.angle(pl_evd), label=f'tau: {tau}')

plt.legend(loc='best')
plt.xlabel('t')
plt.ylabel('Phase [rad]')
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()


'''
   NOTES: TODO

'''
