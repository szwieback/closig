'''
Created on Jan 11, 2023

@author: simon
'''

import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
globfigparams = {
    'fontsize':8, 'family':'serif', 'usetex': True,
    'preamble': r'\usepackage{amsmath} \usepackage{times} \usepackage{mathtools}',
    'column_inch':229.8775 / 72.27, 'markersize':24, 'markercolour':'#AA00AA',
    'fontcolour':'#666666', 'tickdirection':'out', 'linewidth': 0.5,
    'ticklength': 2.50, 'minorticklength': 1.1 }

cols = {'true': '#000000', 'est': '#aa9966', 'unc': '#9999ee'}
colslist = ['#2b2d47', '#8a698c', '#b29274', '#aaaaaa']
cmap_e = cc.cm['bmy']

def initialize_matplotlib():
    plt.rc('font', **{'size':globfigparams['fontsize'], 'family':globfigparams['family']})
    plt.rcParams['text.usetex'] = globfigparams['usetex']
    plt.rcParams['text.latex.preamble'] = globfigparams['preamble']
    plt.rcParams['legend.fontsize'] = globfigparams['fontsize']
    plt.rcParams['font.size'] = globfigparams['fontsize']
    plt.rcParams['axes.linewidth'] = globfigparams['linewidth']
    plt.rcParams['axes.labelcolor'] = globfigparams['fontcolour']
    plt.rcParams['axes.edgecolor'] = globfigparams['fontcolour']
    plt.rcParams['xtick.color'] = globfigparams['fontcolour']
    plt.rcParams['xtick.direction'] = globfigparams['tickdirection']
    plt.rcParams['ytick.direction'] = globfigparams['tickdirection']
    plt.rcParams['ytick.color'] = globfigparams['fontcolour']
    plt.rcParams['xtick.major.width'] = globfigparams['linewidth']
    plt.rcParams['ytick.major.width'] = globfigparams['linewidth']
    plt.rcParams['xtick.minor.width'] = globfigparams['linewidth']
    plt.rcParams['ytick.minor.width'] = globfigparams['linewidth']
    plt.rcParams['ytick.major.size'] = globfigparams['ticklength']
    plt.rcParams['xtick.major.size'] = globfigparams['ticklength']
    plt.rcParams['ytick.minor.size'] = globfigparams['minorticklength']
    plt.rcParams['xtick.minor.size'] = globfigparams['minorticklength']
    plt.rcParams['text.color'] = globfigparams['fontcolour']

def prepare_figure(
        nrows=1, ncols=1, figsize=(1.7, 0.8), figsizeunit='col', sharex='col', sharey='row',
        squeeze=True, bottom=0.10, left=0.15, right=0.95, top=0.95, hspace=0.5, wspace=0.1,
        remove_spines=True, gridspec_kw=None, subplot_kw=None):

    initialize_matplotlib()
    if figsizeunit == 'col':
        width = globfigparams['column_inch']
    elif figsizeunit == 'in':
        width = 1.0
    figprops = dict(facecolor='white', figsize=(figsize[0] * width, figsize[1] * width))
    if nrows > 0 and ncols > 0:
        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols , sharex=sharex, sharey=sharey, squeeze=squeeze,
            gridspec_kw=gridspec_kw, subplot_kw=subplot_kw)
        plt.subplots_adjust(bottom=bottom, left=left, right=right, top=top, hspace=hspace,
                            wspace=wspace)
    else:
        fig = plt.figure()
        axs = None
    fig.set_facecolor(figprops['facecolor'])
    fig.set_size_inches(figprops['figsize'], forward=True)
    if remove_spines:
        try:
            for ax in axs.flatten():
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
        except:
            axs.spines['right'].set_visible(False)
            axs.spines['top'].set_visible(False)
    return fig, axs

def real_closures(cclosures):
    return np.angle(cclosures) * 180 / np.pi

def triangle_plot(
        basis, cclosures, cclosures_ref=None, ax=None, osf=4, aspect=0.75, vabs=None, cmap=None,
        vabs_ref=None, ticks=None, ticklabels=None, show_xticklabels=True, show_yticklabels=True,
        remove_spines=True, blabel=None, plabel=None):
    from scipy.interpolate import griddata
    if ax is None: fig, ax = plt.subplot(1, 1)
    if cmap is None: cmap = cc.cm['CET_C1']
    pt, ptau = basis.pt, basis.ptau
    ptr = np.arange(min(pt), max(pt), step=1 / osf)
    ptaur = np.arange(min(ptau), max(ptau), step=0.5 / osf)
    mg = tuple(np.meshgrid(ptr, ptaur))
    closures = real_closures(cclosures)
    closure_grid = griddata(np.stack((pt, ptau), axis=1), closures, mg, method='nearest')
    closure_grid_linear = griddata(np.stack((pt, ptau), axis=1), closures, mg, method='linear')
    closure_grid[np.isnan(closure_grid_linear)] = np.nan
    if cclosures_ref is not None:
        raise NotImplementedError()
        closures_ref = real_closures(cclosures_ref)
        closure_ref_grid = griddata(np.stack((pt, ptau), axis=1), closures_ref, mg, method='linear')
        assert closures_ref.shape == closures.shape
    if vabs is None: vabs = np.nanmax(np.abs(closure_grid))
    P = basis.P
    extent = (1, P + 0.5, 1, P + 0.5)    
    if ticks is not None: 
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        if ticklabels is not None:
            if show_xticklabels: 
                ax.set_xticklabels(ticklabels)
            else:
                ax.set_xticklabels([])
            if show_yticklabels:
                ax.set_yticklabels(ticklabels)
            else:
                ax.set_yticklabels([])
    from matplotlib.colors import Normalize
    if cclosures_ref is None:
        # ax.imshow(closure_grid[::-1, ...], vmin=-vabs, vmax=vabs, cmap=cmap, aspect=aspect, extent=extent)
        norm = Normalize(-vabs, vabs, clip=True)
        colors = cmap(norm(closure_grid))        
        ax.imshow(
                    colors[::-1, ...], aspect=aspect, extent=extent)        
    else:
        from  matplotlib.colors import rgb_to_hsv, hsv_to_rgb
        raise NotImplementedError()
        norm = Normalize(-3, 3, clip=True)
        colors = cmap(norm(closure_grid / closure_ref_grid))
        colors_hsv = rgb_to_hsv(colors[..., 0:3])
        if vabs_ref is None: vabs_ref = np.nanmean(np.abs(closure_ref_grid))
        closure_ref_grid_clip = np.abs(closure_ref_grid) / vabs_ref
        closure_ref_grid_clip[closure_ref_grid_clip > 1] = 1
        closure_ref_grid_clip[closure_ref_grid_clip < 0] = 0
        colors_hsv[..., 2] *= closure_ref_grid_clip
        colors[..., 0:3] = hsv_to_rgb(colors_hsv)
        ax.imshow(
            colors[::-1, ...], aspect=aspect, extent=extent)
    y_lab = 0.81
    if blabel is not None:
        ax.text(0.98, y_lab, blabel, transform=ax.transAxes, ha='right', va='baseline')
    if plabel is not None:
        from string import ascii_lowercase
        ax.text(0.04, y_lab, ascii_lowercase[plabel] + ')', transform=ax.transAxes, ha='left', va='baseline')        
    ax.set_ylim(extent[2:])
    ax.set_xlim(extent[:2])
    ax.grid(True, axis='both', color='#666666', lw=0.5, alpha=0.1)
    if remove_spines:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)    
    return ax
