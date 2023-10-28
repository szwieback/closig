'''
    How does velocity bias depend on precipitation frequency for an alternative statistical model?
'''
from closig.model import LohmanPrecipLayer
from closig.expansion import SmallStepBasis, TwoHopBasis
from closig.visualization import triangle_plot
from matplotlib import pyplot as plt
from closig.linking import EMI, EMI_py, CutOffRegularizer, NearestNeighbor, EVD
import numpy as np


# Tile shrub layers together

model = LohmanPrecipLayer(f=20, tau=0.5, dcoh=0.8,
                          coh0=0, var=0.5)

# Analysis & Plotting
P = 90
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


taus = [2, 5, 10, 30, 45, P]
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
