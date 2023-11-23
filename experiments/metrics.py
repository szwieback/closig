'''
Created on Nov 21, 2023

@author: simon
'''
from abc import abstractmethod
import numpy as np

class PhaseHistoryMetric():

    @abstractmethod
    def __init__(self,):
        pass

    @abstractmethod
    def evaluate(self, ph1, ph2, **kwargs):
        pass

    @classmethod
    def deviation(cls, ph1, ph2):
        if len(ph1.shape) > len(ph2.shape):
            sl = (None,) * (len(ph1.shape) - len(ph2.shape)) + (Ellipsis,)
            _ph2 = ph2[sl]
        else:
            _ph2 = ph2
        error = ph1 * _ph2.conj()
        error /= np.abs(error)
        return error

class CosMetric(PhaseHistoryMetric):

    def __init__(self):
        pass

    def evaluate(self, ph1, ph2, axis=(0,)):
        error = self.deviation(ph1, ph2)
        return 0.5 * (1 - np.mean(np.real(error), axis=axis))

class MeanDeviationMetric(PhaseHistoryMetric):

    def __init__(self):
        pass

    def evaluate(self, ph1, ph2, axis=(0,)):
        error = self.deviation(ph1, ph2)
        return np.angle(np.mean(error, axis=axis))
    
class TrendSeasonalMetric(PhaseHistoryMetric):
    
    def __init__(self, P_year=None):
        self.P_year = P_year
        
    def predictor_matrix(self, P):
        X = np.zeros((3, P))
        t_y = np.arange(P) / self.P_year
        X[0,:] = t_y
        X[1,:] = np.cos(2 * np.pi * t_y)
        X[2,:] = np.sin(2 * np.pi * t_y)
        return X
    
    def evaluate(self, ph1, ph2):
        #requires (..., P) shape
        dphi = self.deviation(ph1, ph2)
        P = dphi.shape[-1]
        dphi_unw = np.unwrap(np.angle(dphi), axis=-1)
        dphi_unw_rs = np.reshape(dphi_unw, (-1, P))
        X = self.predictor_matrix(P)
        beta = np.linalg.lstsq(X.T, dphi_unw_rs.T, rcond=None)[0]
        return np.reshape(beta.T, dphi_unw.shape[:-1] + (X.shape[0],))         

class PeriodogramMetric(PhaseHistoryMetric):
    
    def __init__(self, P_year):
        self.P_year = P_year
        
    def evaluate(self, ph1, ph2):
        from scipy.signal import periodogram
        from scipy.fft import fftshift
        phd = self.deviation(ph1, ph2)
        P = phd.shape[-1]
        mask = np.any(np.logical_not(np.isfinite(phd)), axis=-1)
        phd[mask, :] = 0
        f, p = periodogram(phd, axis=-1, nfft=2 * P, fs=P_year, return_onesided=False)
        p[mask, :] = np.nan
        f, p = fftshift(f), fftshift(p, axes=-1)
        return f, p
    

if __name__ == '__main__':
    from pathlib import Path
    p0 = Path('/home/simon/Work/closig')
    ps = p0 / 'stacks'
    p1 = p0 / 'processed/phasehistory'
    P_year = 30.4
    fns = list(p1.glob('*.npy'))
    fn = fns[6]
    # fn = fns[0]
    
    ph = np.load(fn)

    tsm = TrendSeasonalMetric(P_year=P_year)
    beta = tsm.evaluate(ph[..., 0, :], ph[..., -1, :])
    
    pm = PeriodogramMetric(P_year=P_year)
    f, psd = pm.evaluate(ph[..., 0, :], ph[..., -1, :])
    psd_avg = np.nanmean(psd, axis=(0, 1))
    print(f)
    
    import matplotlib.pyplot as plt
    plt.semilogy(f, psd_avg)
    
    plt.ylim((1e-3, 1e1))
    plt.show()
    
    
    # fix linker
    # from experiments.experiment import load_C
    # C = load_C(ps / fn.name).reshape((-1,)+ (phd.shape[-1], )*2)
    # # differences negligible    
    # C_diag = np.diagonal(C, offset=-1, axis1=-2, axis2=-1)
    # ph_nn = np.ones(C.shape[:-1], dtype=C_diag.dtype) 
    # print(C.shape, C_diag.shape, ph_nn.shape)
    # ph_nn[:, 1:] = np.cumproduct(C_diag, axis=-1)
    # ph_nn /= np.abs(ph_nn)
    #
    # # phd = ph[..., 0,:] * ph_nn.conj()
    # # phd /= np.abs(phd)
    # # phd = np.reshape(phd, (-1, phd.shape[-1]))
    #
    #
    #
    #
    #
    # t_y = np.arange(phd.shape[-1]) / P_year 
    # plt.plot(t_y, np.angle(phd[::200,:]).T, alpha=0.2)
    # # plt.plot(t_y, np.angle(np.sum(phd, axis=0)))
    # plt.show()
    #
