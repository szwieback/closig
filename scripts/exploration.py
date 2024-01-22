'''
Created on Nov 28, 2023

@author: simon
'''
from pathlib import Path
import numpy as np

from closig import load_object
from closig import SmallStepBasis, TwoHopBasis

def temporal_ml(fnim):
    C_vec = np.moveaxis(np.load(fnim), 0, -1)
    P = int(-0.5 + np.sqrt(0.25 + 2 * C_vec.shape[-1]))
    assert (P * (P+1)) // 2 == C_vec.shape[-1]
    ind = np.tril_indices(P) if lower else np.triu_indices(P)
    sigma = np.zeros(C_vec.shape[:-1], dtype=np.float64)
    for jind in range(C_vec.shape[-1]):
        if ind[0][jind] == ind[1][jind]:
            sigma += np.abs(C_vec[..., jind])
    sigma /= P
    return sigma

def evaluate_basis(C_vec, Basis):
    from greg import extract_P
    P = extract_P(C_vec.shape[-1])
    basis = Basis(P)
    cclosures = basis.evaluate_covariance(C_vec, normalize=True, compl=True, vectorized=True)
    return (basis, cclosures) 


p0 = Path('/home/simon/Work/closig/')
roi = 'Colorado_fields'
metrics = load_object(p0 / 'processed' / 'metrics' / f'{roi}.p')

trendparms = metrics[0]['trend'][0]
trend, amplitude = trendparms[..., 0], np.linalg.norm(trendparms[..., 1:], axis=-1)
print(trend.shape)

fnim = p0 / 'stacks' / f'{roi}.npy'
fnml = p0 / 'processed' / 'ml' / f'{roi}.npy'
lower = False
if not fnml.exists():
    sigma = temporal_ml(fnim)
    np.save(fnml, sigma)
else:
    sigma = np.load(fnml)

# add P to metrics

C_vec = np.moveaxis(np.load(fnim), 0, -1)
Bases = {'small steps': SmallStepBasis, 'two hops': TwoHopBasis}
for bn in Bases:
    Basis = Bases[bn]
    evaluate_basis(C_vec, Basis)

import matplotlib.pyplot as plt
fig, axs = plt.subplots(ncols=3, sharex=True, sharey=True)
axs[0].imshow(trend, cmap='seismic', vmin=-1, vmax=1)
axs[1].imshow(amplitude, cmap='magma', vmin=-1, vmax=1)
axs[2].imshow(10 * np.log10(sigma), cmap='Greys_r')
plt.show()

