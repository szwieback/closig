'''
Created on Nov 18, 2023

@author: simon
'''

import numpy as np
from pathlib import Path

from experiments import CutOffDataExperiment


p0 = Path('/home2/Work/closig/')
# p0 = Path('/home/simon/Work/closig')
pin = p0 / 'stacks'
pout = p0 / 'processed/phasehistory'
dps0, add_full = (1, 31), True
#P_year=30.4

fns = list(pin.glob('*.npy'))

for fn in fns:
    print(fn)
    fout = pout / fn.name    
    ex = CutOffDataExperiment.from_file(fn, dps=dps0, add_full=add_full)
    ph = ex.phase_history(N_jobs=48)
    np.save(fout, ph)
    