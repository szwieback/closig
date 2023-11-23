'''
Created on Nov 18, 2023

@author: simon
'''

from pathlib import Path
from closig import save_object, load_object
from closig.experiments import CutOffDataExperiment, PeriodogramMetric, TrendSeasonalMetric

p0 = Path('/home2/Work/closig/')
# p0 = Path('/home/simon/Work/closig')
pin = p0 / 'stacks'
pout = p0 / 'processed/'
dps, add_full = (1, 16, 31), True
P_year = 30.4
overwrite = False

metrics = {'trend': TrendSeasonalMetric(P_year=P_year), 'psd': PeriodogramMetric(P_year=P_year)}
meta = {'P_year': P_year, 'dps': dps, add_full: add_full}
fns = list(pin.glob('*.npy'))

for fn in fns:
    fnout = (pout / 'phasehistory' / fn.stem).with_suffix('.p')
    if overwrite or not fn.exists():
        ex = CutOffDataExperiment.from_file(fn, dps=dps, add_full=add_full)
        ph = ex.phase_history(N_jobs=48)
        save_object((ph, meta), fnout)
    ph = load_object(fnout)[0]
    res = {}
    for metric in metrics:
        res[metric] = [metrics[metric].evaluate(ph[..., jdp,:], ph[..., -1,:])
                       for jdp, dp in enumerate(dps)]
    save_object((ph, meta), (pout / 'metrics' / fn.stem).with_suffix('.p'))

