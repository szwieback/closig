'''
Created on Nov 6, 2023

@author: simon
'''
import numpy as np
import numpy.ma as ma

from closig.visualization import prepare_figure, initialize_matplotlib, colslist

y_xlab_def = -0.40
x_ylab_def = -0.18

def mask_angle(angle):
    angle_arr = ma.array(angle)
    angle_arr[np.concatenate([(False,), angle[1:] * angle[:-1] < -6])] = ma.masked
    return angle_arr
    
def phase_error_plot(ex, ax=None, show_xlabel=True, show_ylabel=True, y_xlab=None, x_ylab=None):
    if ax is None:
        _, ax = prepare_figure()
    dp, phe = ex.phase_error()
    dp_years = ex._dp_years(dp)
    angle_dp = np.angle(phe)
    mask_angle_dp = mask_angle(angle_dp)
    col = colslist[0]
    ax.axhline(0.0, c=colslist[-1], lw=0.5, alpha=0.5, zorder=2)
    ax.plot(dp_years, mask_angle_dp, c=col, lw=0.5)
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
    return ax

def phase_history_bias_plot(
        ex, ax=None, show_xlabel=True, show_ylabel=True, y_xlab=None, x_ylab=None, cols=None):
    if cols is None: cols = [c for c in colslist]
    if ax is None:
        _, ax = prepare_figure()
    phe = ex.phase_history_error()
    dps = ex.dps
    p_years = ex._dp_years(ex.p)
    ax.axhline(0.0, c=colslist[-1], lw=0.5, alpha=0.5, zorder=2)
    ax.set_ylim((-np.pi, np.pi))
    ax.set_yticks((-np.pi, 0, np.pi))
    ax.set_yticklabels(('$-\\pi$', '0', '$\\pi$'))    
    for jdp, phe_dp in enumerate(phe):
        col = cols[jdp]
        angle_dp = np.angle(phe_dp)
        mask_angle_dp = mask_angle(angle_dp)
        ax.plot(p_years, mask_angle_dp, c=col, label=dps[jdp], lw=0.5)
        ax.plot(
            p_years[mask_angle_dp.mask], angle_dp[mask_angle_dp.mask], linestyle='none', ms=1, marker='o', mec='none', mfc=col)
    if show_xlabel:
        if y_xlab is None: y_xlab = y_xlab_def
        ax.text(0.50, y_xlab, 'time [yr]', transform=ax.transAxes, va='baseline', ha='center')
    if show_ylabel:
        if x_ylab is None: x_ylab = x_ylab_def
        ax.text(
            x_ylab, 0.50, 'bias [rad]', transform=ax.transAxes, va='center', ha='right', rotation=90)
    return ax

def phase_history_metric_plot(
        ex, ax=None, samples=64, show_xlabel=True, show_ylabel=True, y_xlab=None, x_ylab=None, cols=None):
    if cols is None: cols = [c for c in colslist]
    if ax is None:
        _, ax = prepare_figure()
    C_obs = ex.observed_covariance(samples=(samples,))
    phe = ex.phase_history_error(C_obs)
    metric = ex.cos_metric(phe)
    ax.axhline(0.0, c=colslist[-1], lw=0.5, alpha=0.5, zorder=2)
    dps = ex.dps
    p_years = ex._dp_years(ex.p)

    for jdp, m_dp in enumerate(metric):
        col = cols[jdp]
        ax.plot(p_years, m_dp, c=col, label=dps[jdp], lw=0.5)
    ax.set_ylim((0, 1))
    ax.set_yticks((0, 1))
    if show_xlabel:
        if y_xlab is None: y_xlab = y_xlab_def
        ax.text(0.50, y_xlab, 'time [yr]', transform=ax.transAxes, va='baseline', ha='center')
    if show_ylabel:
        if x_ylab is None: x_ylab = x_ylab_def
        ax.text(
            x_ylab, 0.50, 'cos metric [-]', transform=ax.transAxes, va='center', ha='right', rotation=90)
    return ax