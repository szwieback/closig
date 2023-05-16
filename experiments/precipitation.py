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
    f=5, tau=1.5)


# Analysis & Plotting
P = 60
model.plot_n(P)

fig, ax = plt.subplots(nrows=1, ncols=3)
model.plot_matrices(
    P, coherence=False, displacement_phase=False, ax=[ax[0], ax[1]])

# Compute closure phases and plot
smallStep = SmallStepBasis(P)
closure_phases = smallStep.evaluate_covariance(
    model.covariance(P), compl=True, normalize=False)

print(f'Mean Closure Phase: {np.angle(closure_phases).mean()}')
triangle_plot(smallStep, closure_phases,
              ax=ax[2], cmap=plt.cm.seismic, vabs=45)
ax[2].set_title(
    f'Small Step Basis\n mean: {np.round(np.angle(closure_phases).mean(), 3)} rad')
plt.show()

plt.scatter(model.get_baselines(P).flatten(),
            np.angle(model.covariance(P)).flatten())
plt.xlabel('Baseline [t]')
plt.ylabel('Phase [rad]')
plt.show()

taus = [1, 15, 20, 30, 45, 60]

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
plt.grid()
plt.show()


'''
   NOTES: TODO

'''
