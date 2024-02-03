'''
Created on Feb 1, 2024

@author: simon
'''
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.lines as mlines
from matplotlib.colors import Normalize

from closig import load_object
from closig.visualization import prepare_figure, cmap_clipped, cmap_cyclic, colslist, triangle_plot

def read_stack(roi, p0):
    pp = p0 / 'processed'
    res = {}
    metrics, meta_metrics = load_object(pp / 'metrics' / f'{roi}.p')
    res['meta'] = meta_metrics
    res['metrics'] = metrics
    cclosures = load_object(pp / 'cclosures' / f'{roi}.p')
    res['cclosures'] = {
        b: (cclosures[b][0], np.mean(cclosures[b][1], axis=(1, 2))) for b in cclosures}
    ph = load_object(pp / 'phasehistory' / f'{roi}.p')[0]
    res['phdiff'] = np.mean(ph * ph[..., -1,:][..., np.newaxis,:].conj(), axis=(0, 1))
    return res

lamb = 0.055
conv = (lambda phi: phi * lamb / (4 * np.pi), lambda d: 4 * np.pi * d / lamb)

if __name__ == '__main__':
    p0 = Path('/home/simon/Work/closig/')
    fnout = p0 / 'figures' / 'triangle_speckle.pdf'

    rois = ['Colorado_grass', 'Colorado_mountains', 'NewMexico_flat', 'NewMexico_dissected']
    # 'NewMexico_eroded' has seasonal variability in ph
    headers = ['grassland CO', 'mountainous CO', 'semi-arid plain NM', 'dissected NM']

    aspect = 0.75
    triangleparms = [
        (180, cmap_cyclic, ['$-\\pi$', '0', '$\\pi$']),
        (45, cmap_clipped, ['$-\\pi/4$', '0', '$\\pi/4$'])]
    cax_limits = [1.28, 0.1, 0.10, 0.80]
    dps_labels = ['nearest', '1/2 yr', '1 yr']
    fig, axs = prepare_figure(
        nrows=4, ncols=len(rois), figsize=(1.90, 1.05), sharey='row', sharex='col', top=0.96, left=0.06,
        wspace=0.20, hspace=0.18, bottom=0.10, right=0.89)

    for ax in axs[len(triangleparms):,:].flatten():
        ax.set_box_aspect(aspect)

    for jroi, roi in enumerate(rois):
        if jroi >=0:#in (0, 3):
            res = read_stack(roi, p0)
            P_year = res['meta']['P_year']
            for jbn, bn in enumerate(['small steps', 'two hops']):
                b, cclosures = res['cclosures'][bn]
                triangle_plot(
                    b, cclosures, ax=axs[jbn, jroi], vabs=triangleparms[jbn][0], cmap=triangleparms[jbn][1],
                    fs=P_year, cbar=False)
            P = res['phdiff'].shape[-1]
            x_extent = (0, (P - 0.5) / P_year)
            x = np.arange(P) / P_year
            for ax in axs[2:, jroi]:
                ax.axhline(0.0, color='#666666', lw=0.5, alpha=0.1)
            for jdp in (2, 1, 0):
                axs[2, jroi].plot(x, np.angle(res['phdiff'][jdp,:]), c=colslist[jdp], lw=0.6)
            for jdp in (2, 1):
                axs[3, jroi].plot(x, np.angle(res['phdiff'][jdp,:]), c=colslist[jdp], lw=0.6)
            axs[0, jroi].text(
                0.50, 1.07, headers[jroi], c='k', transform=axs[0, jroi].transAxes, ha='center',
                va='baseline')
            axs[-1, jroi].text(0.50, -0.47, '$t$ [yr]', transform=ax.transAxes, va='baseline', ha='center')
    for ax in axs[-1,:]:
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xlim(0.0, 3.0)
    for ax in axs[0:2, 0]:
        ax.set_yticks([0, 1, 2, 3])
    axs[2, 0].set_ylim(-1.86, 1.86)
    axs[3, 0].set_ylim(-np.pi / 8, np.pi / 8)
    for jax, ax in enumerate(axs[0:len(triangleparms), -1]):
        cax = ax.inset_axes(cax_limits)
        vabs = triangleparms[jax][0] / 180 * np.pi
        cbar = plt.colorbar(
            ScalarMappable(norm=Normalize(vmin=-vabs, vmax=vabs), cmap=triangleparms[jax][1]), cax,
            shrink=0.5, orientation='vertical', ticks=[-vabs, 0, vabs])
        cbar.set_ticklabels(triangleparms[jax][2])
    ax = axs[-2, -1]
    ax.set_yticks((-np.pi / 2, 0, np.pi / 2))
    ax.set_yticklabels(('$-\\frac{\\pi}{2}$', '0', '$\\frac{\\pi}{2}$'))
    handles = [mlines.Line2D([], [], color=c, label=l, lw=0.8) for l, c in zip(dps_labels, colslist)]
    ax.legend(
        handles=handles, bbox_to_anchor=(cax_limits[0], -0.11), loc='center left', borderaxespad=0.0,
        frameon=False, borderpad=0.3, handlelength=1.0, handletextpad=0.6)
    secax = ax.secondary_yaxis('right', functions=conv)
    secax.set_yticks((-5e-3, 0, 5e-3))
    secax.set_yticklabels((-5, 0, 5))
    ax = axs[-1, -1]
    ax.set_yticks((-np.pi / 8, 0, np.pi / 8))
    ax.set_yticklabels(('$-\\frac{\\pi}{8}$', '0', '$\\frac{\\pi}{8}$'))
    secax2 = ax.secondary_yaxis('right', functions=conv)
    secax2.set_yticks((-1e-3, 0, 1e-3))
    secax2.set_yticklabels((-1, 0, 1))
    for ax in axs[-2:, -1]:
        ax.text(1.24, 0.50, '[mm]', rotation=270, ha='left', va='center', transform=ax.transAxes)
    ylabels = [
        ('small step', '$\\tau$ [yr]'), ('two hop', '$\\tau$ [yr]'),
        ('$\\phi$ bias', '[rad]'), ('$\\phi$ bias', '[rad]')]
    for jax, ax in enumerate(axs[:, 0]):
        ylabell, ylabelr = ylabels[jax]
        ax.text(-0.41, 0.50, ylabell, ha='right', va='center', transform=ax.transAxes, rotation=90)
        ax.text(-0.26, 0.50, ylabelr, ha='right', va='center', transform=ax.transAxes, rotation=90)

    plt.show()
