'''
Created on Jan 5, 2023

@author: simon
'''
from pathlib import Path
import numpy as np

stackname = 'Colorado'
p0 = Path('/home/simon/Work/closig/stacks')
fn = p0 / f'{stackname}.npy'
overwrite = False

stack = np.load(fn, mmap_mode='r')
D = stack.shape[0]
N = int(1 / 2 + np.sqrt((1 / 4) + 2 * D))
arcs = np.array([(n0, n1) for n0 in range(N) for n1 in range(n0, N)])

rois = {'grass': (230, 300, 260, 320), 'mountains': (330, 420, 850, 930), 'fields': (400, 460, 140, 210),
        'rocky': (90, 160, 520, 580)}

for roi in rois:
    fns = p0 / f'{stackname}_{roi}.npy'
    if not fns.exists() or overwrite:
        pt = rois[roi]
        stack_roi = stack[:, pt[0]:pt[1], pt[2]:pt[3]].copy()
        np.save(fns, stack_roi)



# import matplotlib.pyplot as plt
# intensity = 10 * np.log10(np.abs(stack[0, ...]))
# plt.imshow(intensity, cmap='bone')
# plt.show()
        