from matplotlib import pyplot as plt
import numpy as np
from abc import abstractmethod

from closig import CutOffRegularizer, EVD, TiledModel, HomogSoilLayer, Geom

class Experiment():
    @abstractmethod
    def __init__(self):
        pass
    
class CutOffExperiment(Experiment):
    def __init__(self, model, dps):
        self.dps = dps

P = 90
P_year = 30
geom = Geom(theta=30*np.pi/180, wavelength=0.05)


dz = -0.05 / P_year
coh0 = 0.3 # expect phase linking error to increase with coh0 for tau >> 1
dcoh = 0.6
center = HomogSoilLayer(dz=0, coh0=coh0, dcoh=dcoh)
trough = HomogSoilLayer(dz=dz, coh0=coh0, dcoh=dcoh)
fractions = [0.75, 0.25]
model = TiledModel([center, trough], fractions=fractions, geom=geom)

# error = model.covariance(P, coherence=True, displacement_phase=False) * (
#     model.covariance(P, coherence=True, displacement_phase=True).conj())
# plt.scatter(model.get_baselines(P).flatten(),
#             np.angle(error.flatten()), s=10, alpha=0.5)
# plt.xlabel('Baseline')
# plt.title(f'Phase error')
# plt.ylabel('Error [rad]')
# plt.show()

dps = [1, 15, 30, 45, 60, 90]
colors = plt.cm.viridis(np.linspace(0, 1, len(dps)))
Cd = model.covariance(P, coherence=True, displacement_phase=True)
pld = EVD().link(Cd)
C = model.covariance(P, coherence=True, displacement_phase=False)
print(np.angle(C[0, :5]))
print(np.angle(Cd[0, :5]))
# print(model._covariance_element(0, 1))
# print(model._covariance_element(0, 2))
# print(model._covariance_element(0, 3))
# print(model._covariance_element(0, 4))


for dp, color in zip(dps, colors):
    pl_evd = EVD(CutOffRegularizer(dp)).link(C)
    plt.plot(np.angle(pl_evd * pld.conj()),
             label=f'dp: {dp}', color=color)

plt.legend(loc='best')
plt.xlabel('t')
plt.ylabel('Phase [rad]')
plt.title(f'Phase history relative to displacement-equivalent phase history')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

