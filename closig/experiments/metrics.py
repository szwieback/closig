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
        dphi_rs = np.reshape(dphi, (-1, P))
        mask = np.any(np.logical_not(np.isfinite(dphi_rs)), axis=-1)
        dphi_rs[mask, :] = 1                
        dphi_unw_rs = np.unwrap(np.angle(dphi_rs), axis=-1)
        X = self.predictor_matrix(P)
        beta = np.linalg.lstsq(X.T, dphi_unw_rs.T, rcond=None)[0].T
        beta[mask, :] = np.nan
        return np.reshape(beta, dphi.shape[:-1] + (X.shape[0],))         

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
        f, p = periodogram(phd, axis=-1, nfft=2 * P, fs=self.P_year, return_onesided=False)
        p[mask, :] = np.nan
        f, p = fftshift(f), fftshift(p, axes=-1)
        return f, p
    