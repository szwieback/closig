'''
Created on Jan 11, 2023

@author: simon
'''
from pathlib import Path
import numpy as np

p0 = Path('/home/simon/Work/closig/stacks')
pfig = Path('/home/simon/Work/closig/figures')


from closig import SmallStepBasis, TwoHopBasis
from closig.visualization import triangle_plot, prepare_figure, cmap_clipped, cmap_cyclic
from matplotlib import colors
import colorcet as cc


def plot_triangles(roi, p0, pfig):
    
    fns = p0 / f'{roi}.npy'
    pfig.mkdir(exist_ok=True)
    Cvec = np.moveaxis(np.load(fns), 0, -1)
    N = int(-1 / 2 + np.sqrt((1 / 4) + 2 * Cvec.shape[-1]))
    bases = {'small steps': SmallStepBasis(N), 'two hops': TwoHopBasis(N)}
    vabs = {'small steps': 180, 'two hops': 45}
    cmaps = {'small steps': cmap_cyclic, 'two hops': cmap_clipped}
    fig, axs = prepare_figure(
        ncols=2, figsize=(1.20, 0.44), left=0.10, right=0.87, top=1.00, bottom=0.14, wspace=0.50)
    for jbase, basename in enumerate(bases):
        b, ax = bases[basename], axs[jbase]
        cclosures = b.evaluate_covariance(Cvec, normalize=True, compl=True, vectorized=True)
        triangle_plot(b, np.mean(cclosures, axis=(1, 2)), ax=ax, vabs=vabs[basename], cmap=cmaps[basename])
        ax.text(0.50, 1.05, basename, ha='center', va='baseline', transform=ax.transAxes)
        ax.text(0.50, -0.34, 'time $t$ [scene]', ha='center', va='baseline', transform=ax.transAxes)
    axs[0].text(
        -0.20, 0.50, 'time scale $\\tau$ [scene]', ha='right', va='center', transform=axs[0].transAxes, 
        rotation=90)
    print(pfig / f'{roi}.pdf')
    fig.savefig(pfig / f'{roi}.pdf', dpi=450)
    
if __name__ == '__main__':
    rois = ['Colorado_rocky', 'Colorado_mountains', 'Colorado_grass', 'Colorado_fields']
    # rois = ['NewMexico_dissected', 'NewMexico_flat', 'NewMexico_mountain', 'NewMexico_eroded']    
    for roi in rois:
        plot_triangles(roi, p0, pfig)
