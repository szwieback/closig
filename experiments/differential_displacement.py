from matplotlib import pyplot as plt
import numpy as np

from closig import CutOffRegularizer, EVD, TiledModel, HomogSoilLayer, Geom
from experiments import CutOffExperiment


P = 90
P_year = 30
geom = Geom(theta=30*np.pi/180, wavelength=0.05)


dz = -0.025 / P_year
coh0 = 0.3 # expect phase linking error to increase with coh0 for tau >> 1
dcoh = 0.6
center = HomogSoilLayer(dz=0, coh0=coh0, dcoh=dcoh)
trough = HomogSoilLayer(dz=dz, coh0=coh0, dcoh=dcoh)
fractions = [0.8, 0.2]
model = TiledModel([center, trough], fractions=fractions, geom=geom)

dps = [1, 15, 30, 45, 60, 90]
ex = CutOffExperiment(model, dps=dps, P=P, P_year=P_year)

dp, phe = ex.phase_error()
plt.scatter(dp, np.angle(phe))
plt.xlabel('Baseline')
plt.title(f'Phase error')
plt.ylabel('Error [rad]')
plt.show()


cmap = plt.cm.viridis

phe = np.angle(ex.phase_history_error())
for jdp, phe_dp in enumerate(phe):
    plt.plot(phe_dp, c=cmap(jdp / (len(dps) - 1)), label=dps[jdp])

plt.legend(loc='best')
plt.xlabel('t')
plt.ylabel('Phase [rad]')
plt.title(f'Phase history relative to displacement-equivalent phase history')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

