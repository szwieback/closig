'''
Created on Nov 28, 2023

@author: simon
'''
from pathlib import Path
import numpy as np

from closig import load_object
from closig.experiments import MeanClosureMetric, PSDClosureMetric
from closig.experiments.metrics import MeanClosureMetric

p0 = Path('/home/simon/Work/closig/')
roi = 'Colorado_grass'
metrics = load_object(p0 / 'processed' / 'metrics' / f'{roi}.p')

trendparms = metrics[0]['trend'][0]
trend, amplitude = trendparms[..., 0], np.linalg.norm(trendparms[..., 1:], axis=-1)

sigma = load_object(p0 / 'processed' / 'ml' / f'{roi}.p')
cclosures = load_object(p0 / 'processed' / 'cclosures' / f'{roi}.p')

b, cc = cclosures['small steps']
mcm = MeanClosureMetric(2, tolerance=0.5)
mean_ss = mcm.evaluate(b, cc)

b, cc = cclosures['two hops']  # to indicate inconsistency of long-term interferograms
P_year = metrics[1]['P_year']
psdm = PSDClosureMetric(P_year // 2, P_year, tolerance=0.5, f_tolerance=0.1)
p_enh = psdm.evaluate(b, cc)
print(10 * np.log10(np.mean(p_enh)))
# implement

import matplotlib.pyplot as plt
fig, axs = plt.subplots(ncols=4, sharex=True, sharey=True)
fig.set_size_inches(12, 3)

axs[0].imshow(trend, cmap='seismic', vmin=-1, vmax=1)
axs[1].imshow(-np.angle(mean_ss), cmap='seismic', vmin=-0.3, vmax=0.3, interpolation='nearest')
# axs[1].imshow(amplitude, cmap='Greys_r', vmin=-1, vmax=1)
axs[2].imshow(10 * np.log10(p_enh), cmap='seismic', vmin=-10, vmax=10)
axs[3].imshow(10 * np.log10(sigma), cmap='Greys_r')

# plt.scatter(np.array(b.pt)[ind], np.angle(np.mean(annual, axis=(1, 2))), alpha=0.3)
# p_avg = np.mean(p, axis=(1, 2))
# plt.plot(f, p_avg, alpha=0.3)
plt.show()
