'''
Created on Nov 18, 2023

@author: simon
'''

from pathlib import Path
from closig import save_object, load_object, SmallStepBasis, TwoHopBasis
from closig.experiments import (
    CutOffDataExperiment, PeriodogramMetric, TrendSeasonalMetric, PSDClosureMetric, MeanClosureMetric)

import numpy as np

def temporal_ml(fnim, lower=False):
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

def evaluate_basis(fn, Basis):
    from greg import extract_P
    C_vec = np.moveaxis(np.load(fn), 0, -1)
    P = extract_P(C_vec.shape[-1])
    basis = Basis(P)
    cclosures = basis.evaluate_covariance(C_vec, normalize=True, compl=True, vectorized=True)
    return (basis, cclosures) 

p0 = Path('/home2/Work/closig/')
# p0 = Path('/home/simon/Work/closig')
pin = p0 / 'stacks'
pout = p0 / 'processed/'
dps, add_full = (1, 16, 31), True
P_year = 30.4
overwrite = False

metrics = {'trend': TrendSeasonalMetric(P_year=P_year), 'psd': PeriodogramMetric(P_year=P_year)}
cmetrics = {'mean_2': (MeanClosureMetric(2, tolerance=0.5), 'small steps'),
            'psd_seasonal': (PSDClosureMetric(P_year // 2, P_year, tolerance=0.5, f_tolerance=0.1), 'two hops')}
            # two hops to indicate inconsistency of long-term interferograms
meta = {'P_year': P_year, 'dps': dps, add_full: add_full}
Bases = {'small steps': SmallStepBasis, 'two hops': TwoHopBasis}

fns = list(pin.glob('*.npy'))

for fn in fns:
    print(fn)
    # fnout = (pout / 'phasehistory' / fn.stem).with_suffix('.p')
    # if overwrite or not fn.exists():
    #     ex = CutOffDataExperiment.from_file(fn, dps=dps, add_full=add_full)
    #     ph = ex.phase_history(N_jobs=48)
    #     save_object((ph, meta), fnout)
    # ph = load_object(fnout)[0]
    # res = {}
    # fnmetrics = (pout / 'metrics' / fn.stem).with_suffix('.p')
    # if overwrite or not fnmetrics.exists(): 
    #     for metric in metrics:
    #         res[metric] = [metrics[metric].evaluate(ph[..., jdp,:], ph[..., -1,:])
    #                            for jdp, dp in enumerate(dps)]
    #     save_object((res, meta), fnmetrics)
    fnml = (pout / 'ml' / fn.stem).with_suffix('.p')
    if overwrite or not fnml.exists():
        sigma = temporal_ml(fn, lower=False)
        save_object(sigma, fnml)
    fncclosures = (pout / 'cclosures' / fn.stem).with_suffix('.p')
    res = {}
    if overwrite or not fncclosures.exists():
        for bn in Bases:
            Basis = Bases[bn]
            res[bn] = evaluate_basis(fn, Basis)
        save_object(res, fncclosures)
    else:
        res = load_object(fncclosures)
    fncmetrics = (pout / 'cmetrics' / fn.stem).with_suffix('.p')
    res_cm = {}
    for cmname in cmetrics:
        basisname = cmetrics[cmname][1]
        res_cm[cmname] = cmetrics[cmname][0].evaluate(*res[basisname])
    save_object(res_cm, fncmetrics)
    
    
