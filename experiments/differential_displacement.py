
from closig.model import TiledCovModel, HomogSoilLayer, Geom

from matplotlib import pyplot as plt
from closig.linking import CutOffRegularizer, EVD
import numpy as np

P = 90
P_year = 30
geom = Geom(theta=30*np.pi/180, wavelength=0.05)


dz = -0.15 * geom.wavelength / P_year
coh0 = 0.3 # expect phase linking error to increase with coh0 for tau >> 1
center = HomogSoilLayer(dz=0, coh0=coh0)
trough = HomogSoilLayer(dz=dz, coh0=coh0)
fractions = [0.75, 0.25]
model = TiledCovModel([center, trough], fractions=fractions)

# error = model.covariance(P, coherence=True, displacement_phase=False) * (
#     model.covariance(P, coherence=True, displacement_phase=True).conj())
# plt.scatter(model.get_baselines(P).flatten(),
#             np.angle(error.flatten()), s=10, alpha=0.5)
# plt.xlabel('Baseline')
# plt.title(f'Phase error')
# plt.ylabel('Error [rad]')
# plt.show()

dps = [2, 15, 30, 45, 60, 90]
colors = plt.cm.viridis(np.linspace(0, 1, len(dps)))
pl_true = EVD().link(model.covariance(P, coherence=True, displacement_phase=True))

C = model.covariance(P, coherence=True, displacement_phase=False)
G0 = np.abs(C)

for dp, color in zip(dps, colors):
    pl_evd = EVD(CutOffRegularizer(dp)).link(C)
    plt.plot(np.angle(pl_evd * pl_true.conj()),
             label=f'dp: {dp}', color=color)

plt.legend(loc='best')
plt.xlabel('t')
plt.ylabel('Phase [rad]')
plt.title(f'Phase history relative to displacement-equivalent phase history')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

