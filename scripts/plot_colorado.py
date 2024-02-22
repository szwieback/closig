'''
Created on Nov 28, 2023

@author: simon
'''
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from closig import load_object
from closig.experiments import MeanClosureMetric, PSDClosureMetric
from closig.experiments.metrics import MeanClosureMetric
from closig.visualization import prepare_figure, cmap_div, cmap_mag

def prep(im):
    im2 = im[80:-10, 80:-10, ...]
    return np.moveaxis(im2, 0, 1)

def contrast(imr, percentiles=(5, 95)):
    imr = imr.astype(np.float64)
    lims = np.nanpercentile(imr, percentiles, axis=(0, 1))
    imrr = imr - lims[0,:][np.newaxis, np.newaxis,:]
    imrr /= (lims[1,:] - lims[0,:])[np.newaxis, np.newaxis,:]
    return imrr

def load_corners(rois, stack='Colorado'):
    cmetrics = load_object(p0 / 'processed' / 'cmetrics' / f'{stack}.p')
    shape = cmetrics['mean_2'].shape
    corners_trans = {}
    for roi in rois:
        extent = rois[roi]
        ct = []
        imroi = np.zeros(shape, dtype=np.uint8)
        imroi[extent[0]:extent[1], extent[2]:extent[3]] = 1
        inds_trans = np.nonzero(prep(imroi))
        ct.append((min(inds_trans[0]), min(inds_trans[1])))
        ct.append((max(inds_trans[0]), max(inds_trans[1])))
        corners_trans[roi] = ct
    return corners_trans

def plot_Colorado(p0, fnls, fnout=None):
    from string import ascii_lowercase
    from matplotlib.patches import Rectangle
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize as Nm
    roi = 'Colorado'

    metrics = load_object(p0 / 'processed' / 'metrics' / f'{roi}.p')
    extract_trend = lambda jdp: (
        metrics[0]['trend'][jdp][..., 1], np.linalg.norm(metrics[0]['trend'][jdp][..., 2:], axis=-1))
    trend, amplitude = extract_trend(0)
    trend_s, amplitude_s = extract_trend(1)  # baseline up to 1/2 year.
    cmetrics = load_object(p0 / 'processed' / 'cmetrics' / f'{roi}.p')
    p_enh = cmetrics['psd_qyear']  # Colorado: decentish correspondence to amplitude_s

    imls = contrast(np.load(fnls))

    fig, axs = prepare_figure(
        ncols=7, figsize=(2.05, 0.73), sharex=True, sharey=True, remove_spines=False, left=0.01, right=0.99,
        wspace=0.10, top=0.94, bottom=0.20)
    cmaps = [cmap_div.reversed(), cmap_div, cmap_div, cmap_div.reversed(), cmap_mag, cmap_mag]
    cbarticks = [
        (-np.pi / 4, 0, np.pi / 4), (-np.pi / 4, 0, np.pi / 4), (-np.pi / 12, 0, np.pi / 12), 
        (-np.pi / 24, 0, np.pi / 24), (0, np.pi/6), (1, 5)]
    cbarticklabels = [
        ('-$\\pi/4$', '', '$\\pi/4$'), ('-$\\pi/4$', '', '$\\pi/4$'), ('-$\\pi/12$', '', '$\\pi/12$'), 
        ('-$\\pi/24$', '', '$\\pi/24$'), ('0', '$\\pi/6$'), (1, 5)]
    cbarsecticks = [(-3, 0, 3), (-3, 0, 3), (-1, 0, 1), (-0.5, 0, 0.5), (0.0, 2.0), None]
    cbarsecticklabels = [(-3, '', 3), (-3, '', 3), (-1, '', 1), (-0.5, '', 0.5), None, None]
    cbarunits = ['rad', 'rad/yr', 'rad/yr', 'rad', 'rad', '$-$']
    cbarsecunits = ['mm', 'mm/yr', 'mm/yr', 'mm', 'mm', None]
    norms = [
        Nm(-np.pi / 4, np.pi / 4), Nm(-np.pi / 3, np.pi / 3), Nm(-np.pi / 12, np.pi / 12), 
        Nm(-np.pi / 32, np.pi / 32), Nm(0.0, np.pi/6), Nm(0.5, 5)]
    norms = [
        Nm(-np.pi / 4, np.pi / 4), Nm(-np.pi / 3, np.pi / 3), Nm(-np.pi / 10, np.pi / 10), 
        Nm(-np.pi / 20, np.pi / 20), Nm(0.0, np.pi/6), Nm(0.5, 5)]  
    lamb = 55.0  # mm
    conv = (lambda phi: phi * lamb / (4 * np.pi), lambda d: 4 * np.pi * d / lamb)

    labels = ['$\\bar{\\Xi}_{\\mathrm{ss}}(1\\,\\mathrm{yr})$', '$\\Delta \\beta$ nearest',
              '$\\Delta \\beta$ $1/2$ yr', '$\\bar{\\Xi}_{\\mathrm{th}}(1\\,\\mathrm{yr})$',
              '$\\alpha$ $1/2$ yr', '$p_r(1\\,\\mathrm{yr})$', 'Landsat']
    ipl = 'gaussian'
    axs[0].imshow(prep(np.angle(cmetrics['mean_year'])), cmap=cmaps[0], norm=norms[0], interpolation=ipl)
    axs[1].imshow(prep(trend), cmap=cmaps[1], norm=norms[1], interpolation=ipl)
    axs[2].imshow(prep(trend_s), cmap=cmaps[2], norm=norms[2], interpolation=ipl)
    axs[3].imshow(
        prep(np.angle(cmetrics['mean_year_th'])), cmap=cmaps[3], norm=norms[3], interpolation=ipl)
    axs[4].imshow(prep(amplitude_s), cmap=cmaps[4], norm=norms[4])
    axs[5].imshow(prep(p_enh), cmap=cmaps[5], norm=norms[5])
    axs[6].imshow(imls)

    from scripts.colorado import rois
    corners = load_corners(rois, stack=roi)
    ax = axs[6]
    for isn in ('mountains', 'grass'):
        extent = corners[isn]
        corner = (extent[0][1], extent[0][0])
        w, h = extent[1][1] - extent[0][1], extent[1][0] - extent[0][0]
        ax.add_patch(
            Rectangle(corner, w, h, fill=True, ec='w', lw=1, fc='none', zorder=6))
        ax.text(corner[0] - 15, corner[1] + h / 2, isn[0], ha='right', va='center', c='w',
                bbox={'fc': '#333333', 'ec': 'none', 'alpha': 0.5, 'pad': 0.0})

    for jax, ax in enumerate(axs[:-1]):
        cax = ax.inset_axes([0.10, -0.16, 0.50, 0.05], transform=ax.transAxes)
        mappable = ScalarMappable(norm=norms[jax], cmap=cmaps[jax])
        cbar = plt.colorbar(mappable, ax=ax, cax=cax, orientation='horizontal')
        cbar.solids.set_rasterized(True)
        cbar.solids.set_edgecolor('face')
        cbar.set_ticks(cbarticks[jax])
        cbar.set_ticklabels(cbarticklabels[jax])
        cax.text(1.15, -0.50, f'[{cbarunits[jax]}]', ha='left', va='baseline', transform=cax.transAxes)
        if cbarsecticks[jax] is not None:
            cax2 = cax.secondary_xaxis('top', functions=conv)  # twiny()
            cax2.set_ticks(cbarsecticks[jax])
            cax2.xaxis.set_tick_params(pad=0)
            if cbarsecticklabels[jax] is not None:
                cax2.set_xticklabels(cbarsecticklabels[jax])
            cax.text(
                1.15, 0.90, f'[{cbarsecunits[jax]}]', ha='left', va='baseline', transform=cax.transAxes)

    for jax, ax  in enumerate(axs):
        ax.set_xticks([])
        ax.set_yticks([])
        lab = ascii_lowercase[jax] + ') ' + labels[jax]
        ax.text(0.00, 1.02, lab, ha='left', va='baseline', transform=ax.transAxes,)
    # scale
    ax = axs[-1]
    from matplotlib.lines import Line2D
    line = Line2D(
        (0.58, 0.80), (-0.07,) * 2, lw=0.8, color='#666666', transform=ax.transAxes)
    line.set_clip_on(False)
    ax.add_line(line)
    ax.text(0.69, -0.16, '10 km', ha='center', va='baseline', transform=ax.transAxes)
    if fnout is None:
        plt.show()
    else:
        fig.savefig(fnout, dpi=300)
# add unit labels

if __name__ == '__main__':
    p0 = Path('/home/simon/Work/closig/')
    fnls = p0 / 'optical/Colorado/LC08_L2SP_034032_20200702_20200913_02_T1_resampled.npy'
    fnout = p0 / 'figures' / 'Colorado.pdf'

    plot_Colorado(p0, fnls, fnout=fnout)

