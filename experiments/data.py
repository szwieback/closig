'''
Created on Nov 18, 2023

@author: simon
'''

import numpy as np
from pathlib import Path

from experiments import CutOffExperiment


class CutOffDataExperiment(CutOffExperiment):
    def __init__(self, C, dps, P_year=30):
        self.dps = dps
        self.P = C.shape[-1]
        self.shape = C.shape
        self.C = np.reshape(C, (-1, self.P, self.P))
        self.P_year = P_year
        
    @property
    def Cd(self):
        raise ValueError("Cd not defined for CuttOffDataExperiment")

def load_C(fn):
    from greg import assemble_tril
    Cvec = np.moveaxis(np.load(fn), 0, -1)
    C = assemble_tril(Cvec, lower=False)
    return C

if __name__ == '__main__':
    p0 = Path('/home/simon/Work/closig/stacks')
    C = load_C(p0 / 'NewMexico_eroded.npy')
    ex = CutOffDataExperiment(C, dps=(1, 30))
    ph = ex.phase_history()
    print(ph.shape)
    
    