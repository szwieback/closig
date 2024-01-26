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
cmetrics = load_object(p0 / 'processed' / 'cmetrics' / f'{roi}.p')
print(cmetrics.keys())
mean_ss = cmetrics['mean_year']
p_enh = cmetrics['psd_year']
print(np.max(np.abs(cmetrics['psd_year']-cmetrics['psd_hyear'])))
print(np.max(np.abs(cmetrics['mean_year']-cmetrics['mean_2'])))


import matplotlib.pyplot as plt
fig, axs = plt.subplots(ncols=5, sharex=True, sharey=True)
fig.set_size_inches(12, 3)

axs[0].imshow(trend, cmap='seismic', vmin=-1, vmax=1)
axs[1].imshow(-np.angle(mean_ss), cmap='seismic', vmin=-0.3, vmax=0.3, interpolation='nearest')
axs[2].imshow(amplitude, cmap='Greys_r', vmin=-1, vmax=1)
axs[3].imshow(10 * np.log10(p_enh), cmap='seismic', vmin=-10, vmax=10)
axs[4].imshow(10 * np.log10(sigma), cmap='Greys_r')

# plt.scatter(np.array(b.pt)[ind], np.angle(np.mean(annual, axis=(1, 2))), alpha=0.3)
# p_avg = np.mean(p, axis=(1, 2))
# plt.plot(f, p_avg, alpha=0.3)
plt.show()
