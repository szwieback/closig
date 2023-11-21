'''
Created on Nov 21, 2023

@author: simon
'''
from abc import abstractmethod
import numpy as np

class PhaseHistoryMetric():
    
    @abstractmethod
    def __init__(self, ):
        pass
    
    @abstractmethod
    def evaluate(self, ph1, ph2, axis=(0,)):
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

def MeanDeviationMetric(PhaseHistoryMetric):
    
    def __init__(self):
        pass
    
    def evaluate(self, ph1, ph2, axis=(0,)):
        error = self.deviation(ph1, ph2)
        return np.angle(np.mean(error, axis=axis))    
    


