from pathlib import Path
import numpy as np

stackname = 'Wevok'
p0 = Path('/home/simon/Work/closig/stacks')
fn = p0 / f'{stackname}.npy'
overwrite = False

stack = np.load(fn, mmap_mode='r')
D = stack.shape[0]
N = int(1 / 2 + np.sqrt((1 / 4) + 2 * D))
arcs = np.array([(n0, n1) for n0 in range(N) for n1 in range(n0, N)])

rois = {'hillslope': (270, 310, 850, 970), 'mountains': (235, 275, 580, 650), 
        'toeslope': (160, 220, 390, 460), 'rolling': (180, 220, 870, 950)}

for roi in rois:
    fns = p0 / f'{stackname}_{roi}.npy'
    if not fns.exists() or overwrite:
        pt = rois[roi]
        stack_roi = stack[:, pt[0]:pt[1], pt[2]:pt[3]].copy()
        np.save(fns, stack_roi)


# import matplotlib.pyplot as plt
# im = stack[-1, ...].copy()
# pt = (180, 220, 870, 950)
# im[pt[0]:pt[1], pt[2]:pt[3]] = 1e-6
# intensity = 10 * np.log10(np.abs(im))
# plt.imshow(intensity, cmap='bone')
# plt.show()

