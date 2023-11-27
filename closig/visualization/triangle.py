import numpy as np
import colorcet as cc
import matplotlib.pyplot as plt

def real_closures(cclosures):
    return np.angle(cclosures) * 180 / np.pi

from closig.visualization.plotting import cmap_cyclic

def triangle_plot(
        basis, cclosures, cclosures_ref=None, ax=None, osf=4, aspect=0.75, vabs=None, cmap=None,
        vabs_ref=None, ticks=None, ticklabels=None, show_xticklabels=True, show_yticklabels=True,
        remove_spines=True, blabel=None, plabel=None, show_labels=False, cbar=True, extent=None):
    from scipy.interpolate import griddata
    if ax is None:
        fig, ax = plt.subplot(1, 1)
    if cmap is None:
        cmap = cmap_cyclic
    pt, ptau = basis.pt, basis.ptau
    ptr = np.arange(min(pt), max(pt), step=1 / osf)
    ptaur = np.arange(min(ptau), max(ptau), step=0.5 / osf)
    mg = tuple(np.meshgrid(ptr, ptaur))
    closures = real_closures(cclosures)
    closure_grid = griddata(np.stack((pt, ptau), axis=1),
                            closures, mg, method='nearest')
    closure_grid_linear = griddata(
        np.stack((pt, ptau), axis=1), closures, mg, method='linear')
    closure_grid[np.isnan(closure_grid_linear)] = np.nan
    if cclosures_ref is not None:
        raise NotImplementedError()
        closures_ref = real_closures(cclosures_ref)
        closure_ref_grid = griddata(
            np.stack((pt, ptau), axis=1), closures_ref, mg, method='linear')
        assert closures_ref.shape == closures.shape
    if vabs is None:
        vabs = np.nanmax(np.abs(closure_grid))
    P = basis.P
    if extent is None: extent = (1, P + 0.5, 1, P + 0.5)
    if ticks is not None:
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
    if ticklabels is not None:
        if show_xticklabels:
            ax.set_xticklabels(ticklabels)
        if show_yticklabels:
            ax.set_yticklabels(ticklabels)
    if not show_xticklabels:
        ax.set_xticklabels([])
    if not show_yticklabels:
        ax.set_yticklabels([])
    
    from matplotlib.colors import Normalize
    if cclosures_ref is None:
        norm = Normalize(-vabs, vabs, clip=True)
        colors = cmap(norm(closure_grid))
        im = ax.imshow(closure_grid[::-1, ...], cmap=cmap, norm=norm,
                       aspect=aspect, extent=extent, interpolation='none')
    else:
        from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
        raise NotImplementedError()
        norm = Normalize(-3, 3, clip=True)
        colors = cmap(norm(closure_grid / closure_ref_grid))
        colors_hsv = rgb_to_hsv(colors[..., 0:3])
        if vabs_ref is None:
            vabs_ref = np.nanmean(np.abs(closure_ref_grid))
        closure_ref_grid_clip = np.abs(closure_ref_grid) / vabs_ref
        closure_ref_grid_clip[closure_ref_grid_clip > 1] = 1
        closure_ref_grid_clip[closure_ref_grid_clip < 0] = 0
        colors_hsv[..., 2] *= closure_ref_grid_clip
        colors[..., 0:3] = hsv_to_rgb(colors_hsv)
        ax.imshow(
            colors[::-1, ...], aspect=aspect, extent=extent)
    y_lab = 0.97
    if blabel is not None:
        ax.text(0.98, y_lab, blabel, transform=ax.transAxes,
                ha='right', va='top')
    if plabel is not None:
        from string import ascii_lowercase
        ax.text(0.04, y_lab, ascii_lowercase[plabel] + ')',
                transform=ax.transAxes, ha='left', va='top')
    ax.set_ylim(extent[2:])
    ax.set_xlim(extent[:2])
    ax.grid(True, axis='both', color='#666666', lw=0.5, alpha=0.1)
    if remove_spines:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    if show_labels:
        ax.set_xlabel('t')
        ax.set_ylabel(r'$\\tau$')

    if cbar:
        cax = ax.inset_axes([1.04, 0, 0.05, 1], transform=ax.transAxes)
        _cbar = plt.colorbar(im, ax=ax, cax=cax, ticks=[vabs, 0, -vabs])
        _cbar.solids.set_rasterized(True)
        _cbar.solids.set_edgecolor('face')
    return ax