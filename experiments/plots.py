'''
Created on Nov 6, 2023

@author: simon
'''
import numpy as np
import numpy.ma as ma

from closig.visualization import prepare_figure, initialize_matplotlib, colslist, triangle_plot

y_xlab_def = -0.40
x_ylab_def = -0.18

def mask_angle(angle):
    angle_arr = ma.array(angle)
    angle_arr[np.concatenate([(False,), angle[1:] * angle[:-1] < -6])] = ma.masked
    return angle_arr

def phase_error_plot(
        ex, ax=None, show_xlabel=True, show_ylabel=True, y_xlab=None, x_ylab=None, show_xticklabels=True):
    if ax is None:
        _, ax = prepare_figure()
    dp, phe = ex.phase_error()
    dp_years = ex._dp_years(dp)
    angle_dp = np.angle(phe)
    mask_angle_dp = mask_angle(angle_dp)
    col = colslist[0]
    ax.axhline(0.0, c='#666666', lw=0.5, alpha=0.1, zorder=2)
    ax.plot(dp_years, mask_angle_dp, c=col, lw=0.8)
    ax.plot(
        dp_years[mask_angle_dp.mask], angle_dp[mask_angle_dp.mask], linestyle='none', ms=1, marker='o',
        mec='none', mfc=col)

    ax.set_ylim((-np.pi, np.pi))
    ax.set_yticks((-np.pi, 0, np.pi))
    ax.set_yticklabels(('$-\\pi$', '0', '$\\pi$'))
    if show_xlabel:
        if y_xlab is None: y_xlab = y_xlab_def
        ax.text(0.50, y_xlab, 'baseline [yr]', transform=ax.transAxes, va='baseline', ha='center')
    if show_ylabel:
        if x_ylab is None: x_ylab = x_ylab_def
        ax.text(
            x_ylab, 0.50, 'phase error [rad]', transform=ax.transAxes, va='center', ha='right', rotation=90)
    if not show_xticklabels:
        ax.set_xticklabels([])
    return ax

def phase_history_bias_plot(
        ex, ax=None, show_xlabel=True, show_ylabel=True, y_xlab=None, x_ylab=None, cols=None, 
        show_xticklabels=True):
    if cols is None: cols = [c for c in colslist]
    if ax is None:
        _, ax = prepare_figure()
    phe = ex.phase_history_error()
    dps = ex.dps
    p_years = ex._dp_years(ex.p)
    ax.axhline(0.0, c='#666666', lw=0.5, alpha=0.1, zorder=2)
    ax.set_ylim((-np.pi, np.pi))
    ax.set_yticks((-np.pi, 0, np.pi))
    ax.set_yticklabels(('$-\\pi$', '0', '$\\pi$'))
    for jdp, phe_dp in enumerate(phe):
        col = cols[jdp]
        angle_dp = np.angle(phe_dp)
        mask_angle_dp = mask_angle(angle_dp)
        ax.plot(p_years, mask_angle_dp, c=col, label=dps[jdp], lw=0.8)
        ax.plot(
            p_years[mask_angle_dp.mask], angle_dp[mask_angle_dp.mask], linestyle='none', ms=1, marker='o', 
            mec='none', mfc=col)
    if show_xlabel:
        if y_xlab is None: y_xlab = y_xlab_def
        ax.text(0.50, y_xlab, 'time [yr]', transform=ax.transAxes, va='baseline', ha='center')
    if show_ylabel:
        if x_ylab is None: x_ylab = x_ylab_def
        ax.text(
            x_ylab, 0.50, 'bias [rad]', transform=ax.transAxes, va='center', ha='right', rotation=90)
    if not show_xticklabels:
        ax.set_xticklabels([])        
    return ax

def phase_history_metric_plot(
        ex, ax=None, samples=64, show_xlabel=True, show_ylabel=True, y_xlab=None, x_ylab=None, cols=None,
        show_xticklabels=True):
    if cols is None: cols = [c for c in colslist]
    if ax is None:
        _, ax = prepare_figure()
    C_obs = ex.observed_covariance(samples=(samples,))
    phe = ex.phase_history_error(C_obs)
    metric = ex.cos_metric(phe)
    ax.axhline(0.0, c='#666666', lw=0.5, alpha=0.1, zorder=2)
    dps = ex.dps
    p_years = ex._dp_years(ex.p)

    for jdp, m_dp in enumerate(metric):
        col = cols[jdp]
        ax.plot(p_years, m_dp, c=col, label=dps[jdp], lw=0.8)
    ax.set_ylim((0, 1))
    ax.set_yticks((0, 1))
    if show_xlabel:
        if y_xlab is None: y_xlab = y_xlab_def
        ax.text(0.50, y_xlab, 'time [yr]', transform=ax.transAxes, va='baseline', ha='center')
    if show_ylabel:
        if x_ylab is None: x_ylab = x_ylab_def
        ax.text(
            x_ylab, 0.50, 'cos metric [-]', transform=ax.transAxes, va='center', ha='right', rotation=90)
    if not show_xticklabels:
        ax.set_xticklabels([])      
    return ax

def triangle_experiment_plot(
        ex, Basis, ax=None, show_xlabel=True, show_ylabel=True, y_xlab=None, x_ylab=None, extent=None,
        cmap=None, vabs=180, show_xticklabels=True, show_yticklabels=True, aspect=0.75, ticks=None):
    if ax is None:
        _, ax = prepare_figure()
    basis, cclosures = ex.basis_closure(Basis, compl=True)
    if extent is None: extent = (0, (ex.P + 1)/ex.P_year) * 2
    triangle_plot(
        basis, cclosures, ax=ax, show_xticklabels=show_xticklabels,
         show_yticklabels=True, vabs=vabs, cmap=cmap, cbar=False, aspect=aspect, extent=extent, ticks=ticks)
