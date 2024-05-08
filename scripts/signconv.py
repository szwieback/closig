'''
Created on Feb 15, 2024

@author: simon
'''

# check sign convention

from pathlib import Path
from closig import load_C, NearestNeighborLinker, EVDLinker
import numpy as np

p0 = Path('/home/simon/Work/closig/stacks')

rc0 = (259, 640)#(224, 743,)
rc1 = (251, 682)

C = load_C(p0 / 'kivalina_2019.npy')
print(C.shape)
Cij = C[..., 5, 6]
print(np.angle(Cij[rc1] * Cij[rc0].conj()))

# positive: 5, 6; subsidence; same as model
