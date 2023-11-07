'''
    How does velocity bias depend on precipitation frequency?
'''
from closig.model import PrecipScatterSoilLayer
from closig.expansion import SmallStepBasis, TwoHopBasis
from closig.visualization import triangle_plot
from matplotlib import pyplot as plt
from closig.linking import EMI, EMI_py, CutOffRegularizer, NearestNeighbor, EVD
import numpy as np


# Tile shrub layers together

model = PrecipScatterSoilLayer(
    f=10, tau=1, dcoh=0.8, coh0=0.2, offset=0.1, scale=0.1)


# Analysis & Plotting
P = 60
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
print(f'Mean Closure Phase: {np.angle(c_phases_ss).mean()}')
print(c_phases_ss.shape)

c_phases_th = twoHop.evaluate_covariance(
    model.covariance(P), compl=True, normalize=False)

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


taus = [2, 5, 8, 10, 15, 20, 25, 30, 45, 60]
colors = plt.cm.viridis(np.linspace(0, 1, len(taus)))

for tau, color in zip(taus, colors):

    G = np.abs(model.covariance(P, coherence=True))
    G = CutOffRegularizer().regularize(G, tau_max=tau)
    cov = model.covariance(P, coherence=True)
    cov = cov * G
    pl_evd = EVD().link(cov, G=G)
    plt.plot(np.angle(pl_evd), label=f'tau: {tau}', color=color)


plt.legend(loc='best')
plt.xlabel('t')
plt.ylabel('Phase [rad]')
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()


'''
   NOTES: TODO

'''
