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
roi = 'Colorado'
metrics = load_object(p0 / 'processed' / 'metrics' / f'{roi}.p')

print(len(metrics[0]['trend']))
extract_trend = lambda jdp: (
    metrics[0]['trend'][jdp][..., 0], np.linalg.norm(metrics[0]['trend'][jdp][..., 1:], axis=-1))
trend, amplitude = extract_trend(0)
trend_s, amplitude_s = extract_trend(1) # baseline up to 1/2 year.

sigma = load_object(p0 / 'processed' / 'ml' / f'{roi}.p')
cmetrics = load_object(p0 / 'processed' / 'cmetrics' / f'{roi}.p')
mean_ss = cmetrics['mean_year']

p_enh = cmetrics['psd_qyear'] #COlorado: decentish correspondence to amplitude_s

import matplotlib.pyplot as plt
fig, axs = plt.subplots(ncols=6, sharex=True, sharey=True)
fig.set_size_inches(12, 3)



axs[0].imshow(trend, cmap='seismic', vmin=-1, vmax=1)
axs[1].imshow(-np.angle(mean_ss), cmap='seismic', vmin=-3, vmax=3)
axs[2].imshow(trend_s, cmap='seismic', vmin=-1, vmax=1)
axs[3].imshow(amplitude_s, cmap='Greys_r', vmin=0, vmax=0.2)
axs[4].imshow(10 * np.log10(p_enh), cmap='seismic', vmin=-10, vmax=10)
axs[5].imshow(10 * np.log10(sigma), cmap='Greys_r', vmin=30, vmax=40)

plt.show()
