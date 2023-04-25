'''
Created on Jan 5, 2023

@author: simon
'''
from pathlib import Path
import numpy as np

stackname = 'NewMexico'
p0 = Path('/home/simon/Work/closig/stacks')
fn = p0 / f'{stackname}.npy'
overwrite = False

stack = np.load(fn, mmap_mode='r')
D = stack.shape[0]
N = int(1 / 2 + np.sqrt((1 / 4) + 2 * D))
arcs = np.array([(n0, n1) for n0 in range(N) for n1 in range(n0, N)])

rois = {'dissected': (300, 370, 850, 910), 'flat': (70, 130, 490, 560), 'mountain': (170, 230, 750, 820),
        'eroded': (110, 170, 870, 920)}

for roi in rois:
    fns = p0 / f'{stackname}_{roi}.npy'
    if not fns.exists() or overwrite:
        pt = rois[roi]
        stack_roi = stack[:, pt[0]:pt[1], pt[2]:pt[3]].copy()
        np.save(fns, stack_roi)



# import matplotlib.pyplot as plt
# print(stack.shape)
# # intensity = 10 * np.log10(np.abs(stack[0, ..., ::-1]))
# intensity = 10 * np.log10(np.abs(stack[0, ...]))
# plt.imshow(intensity, cmap='bone')
# plt.show()
#
