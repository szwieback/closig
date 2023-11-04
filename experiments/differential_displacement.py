from matplotlib import pyplot as plt
import numpy as np

from closig import CutOffRegularizer, EVD, TiledModel, HomogSoilLayer, Geom
from experiments import CutOffExperiment, model_catalog


P = 90
P_year = 30
dps = [1, 15, 30, 45, 60, 90]

model = model_catalog('diffdisp', P_year=P_year, band='C')

ex = CutOffExperiment(model, dps=dps, P=P, P_year=P_year)

# dp, phe = ex.phase_error()
# plt.scatter(dp, np.angle(phe))
# plt.xlabel('Baseline')
# plt.title(f'Phase error')
# plt.ylabel('Error [rad]')
# plt.show()


cmap = plt.cm.viridis

# phe = np.angle(ex.phase_history_error())
# for jdp, phe_dp in enumerate(phe):
#     plt.plot(phe_dp, c=cmap(jdp / (len(dps) - 1)), label=dps[jdp])
#
# plt.legend(loc='best')
# plt.xlabel('t')
# plt.ylabel('Phase [rad]')
# plt.grid(alpha=0.2)
# plt.tight_layout()
# plt.show()


C_obs = ex.observed_covariance(samples=(64,))
print(C_obs.shape)
phe = ex.phase_history_error(C_obs)
metric = ex.cos_metric(phe)
for jdp, m_dp in enumerate(metric):
    plt.plot(m_dp, c=cmap(jdp / (len(dps) - 1)), label=dps[jdp])
plt.legend(loc='best')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

