'''
Created on Jan 20, 2023

@author: simon
'''
from abc import abstractmethod

from greg import correlation, force_doubly_nonnegative
import numpy as np

def restrict_kwargs(fun, kwargs):
    rkwargs = {k: v for k, v in kwargs.items() if k in fun.__code__.co_varnames}
    return rkwargs

class Regularizer():
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def regularize(self, G, inplace=False, **kwargs):
        pass

    @classmethod
    def distance_from_diagonal(cls, P):
        mg = np.mgrid[0:P, 0:P]
        return np.abs(mg[1] - mg[0])

class IdleRegularizer(Regularizer):
    def __init__(self):
        pass

    def regularize(self, G, inplace=False):
        _G = G.copy() if not inplace else G
        return _G

class CutOffRegularizer(Regularizer):
    def __init__(self):
        pass

    def regularize(self, G, inplace=False, tau_max=None):
        _G = G.copy() if not inplace else G
        if tau_max is not None:
            slicetup = (None,) * (len(_G) - 2) + (Ellipsis,)
            if _G.shape[-2] != _G.shape[-1]: 
                raise ValueError(f"Expected G of shape (..., P, P) but got {_G.shape}")
            _G *= (self.distance_from_diagonal(_G.shape[-1]) <= tau_max)[slicetup]
        return _G

class Linker():   
    @abstractmethod
    def __init__(self):
        pass

    def test_square(self, C):
        C_shape = C.shape
        P = C_shape[-1]
        if C_shape[-2] != P: 
            raise ValueError(f"Expected G of shape (..., P, P) but got {C.shape}")

    def link(self, C_obs, G=None, corr=True, **kwargs):
        self.test_square(C_obs)
        P = C_obs.shape[-1]
        _C_obs = C_obs
        if corr:
            _C_obs = correlation(C_obs)
        if G is None:
            G = self.estimate_G(C_obs, **restrict_kwargs(self.estimate_G, kwargs))
        else:
            if G.shape != _C_obs.shape: raise ValueError("G needs to have same shape as C_obs")
        shape3 = (-1, P, P)
        cphase = self._link(
            np.reshape(C_obs, shape3), np.reshape(G, shape3), **restrict_kwargs(self.link, kwargs))
        return np.reshape(cphase, C_obs.shape[:-1])

    @abstractmethod
    def _link(self, C_obs, G):
        # C_obs, G of shape (N, P, P)
        pass

    def estimate_G(self, C_obs, tau_max=None):
        G = np.abs(C_obs).real
        if tau_max is not None:
            self._regularize_G(G, tau_max=tau_max)
        G = force_doubly_nonnegative(G, inplace=True)
        return G

class EMI(Linker):
    def __init__(self, regularizer=None):
        if regularizer is not None:
            self.regularizer = regularizer
        else:
            self.regularizer = IdleRegularizer()

    def _regularize_G(self, G, **kwargs):
        return self.regularizer.regularize(G, **kwargs)
    
    def _link(self, C_obs, G):
        from greg import EMI as _EMI
        return _EMI(C_obs, G=G, corr=False)

if __name__ == '__main__':
    C_obs = np.ones((2, 3, 10, 10), dtype=np.complex128)
    emi = EMI(regularizer=CutOffRegularizer())
    print(emi.estimate_G(C_obs, tau_max=4).shape)
    emi.link(C_obs, tau_max=4)
    # print(EMI.distance_from_diagonal(C_obs.shape[-1]) < 3)
