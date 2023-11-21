'''
Created on Jan 20, 2023

@author: simon

'''
from abc import abstractmethod
import closig.graphs as graphs
from greg import correlation, force_doubly_nonnegative
import numpy as np


def restrict_kwargs(fun, kwargs):
    rkwargs = {k: v for k, v in kwargs.items(
    ) if k in fun.__code__.co_varnames}
    return rkwargs


class Subsetter():
    '''
        Subset a modeled covariance matrix given a maximum temporal baseline, or random sampling based 
        on a graph representation of the InSAR network. This simulates an SBAS scenario where temporal baseline dependent velocity errors may arrise. 

        This is a binary and stack-wise operation, thus it is distinct from the regularizer.

        Not working yet, but the idea is to genrate random adjaceny matrices to test whether redunancy 
        is key driver of velocity biases.
    '''

    def __init__(self, P, max_tau=5):
        self.P = P
        self.max_tau = max_tau

    def subset_random(self, G, seed=None):
        '''
            Subset network based on a random graph/adjacency matrix
            returns instance of self
        '''
        k = np.random.uniform(high=1, low=0.5, size=1, seed=seed)
        l = np.random.uniform(high=self.P-2, low=1, size=1, seed=seed)
        G, A = graphs.get_rand_graph(self.P, k=k, l=l)
        self.G = G
        return self

    def G(self):
        return self.G

    def get_cycle_rank(self):
        return graphs.cycle_rank(self.G)


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
    '''
        No regularization.
    '''

    def __init__(self):
        pass

    def regularize(self, G, inplace=False):
        _G = G.copy() if not inplace else G
        return _G


class CutOffRegularizer(Regularizer):
    '''
        Simulate an SBAS network with bandwith/cutoff dp_cutoff (dp: scene difference)
    '''

    def __init__(self, dp_cutoff=None, enforce_dnn=True):
        self.dp_cutoff = dp_cutoff
        self.enforce_dnn = enforce_dnn

    def regularize(self, G, inplace=False):
        if self.dp_cutoff is None:
            _G = G
        else:
            _G = G.copy().astype(np.float64) if not inplace else G
            if _G.shape[-2] != _G.shape[-1]:
                raise ValueError(
                    f"Expected G of shape (..., P, P) but got {_G.shape}")
            mask = (self.distance_from_diagonal(
                _G.shape[-1]) <= self.dp_cutoff)
            _G *= mask
            if self.enforce_dnn: 
                force_doubly_nonnegative(_G, inplace=True)
        return _G


class Linker():
    @abstractmethod
    def __init__(self):
        self.regularizer = IdleRegularizer()

    def link(self, C_obs, G=None, corr=True, N_jobs=1, **kwargs):
        P = C_obs.shape[-1]
        if C_obs.shape[-2] != P: raise ValueError(f"Expected G of shape (..., P, P) but got {C_obs.shape}")
        if G is not None and G.shape != C_obs.shape: raise ValueError("G and C_obs need to have same shape")
        _C_obs = np.reshape(C_obs, (-1, P, P))
        if corr: _C_obs = correlation(C_obs, inplace=True)
        if N_jobs in (0, 1, None):
            cphase = self._link_batch(_C_obs, G, **kwargs)
        else:
            from joblib import Parallel, delayed
            C_obs_split = np.array_split(_C_obs, N_jobs)
            G_split = np.array_split(G, N_jobs) if G is not None else (None, ) * N_jobs
            def _link_single(n): return self._link_batch(C_obs_split[n], G_split[n], **kwargs)
            cphase = np.concatenate(
                Parallel(n_jobs=N_jobs)(delayed(_link_single)(n) for n in range(N_jobs)))
        return np.reshape(cphase, C_obs.shape[:-1])

    def _link_batch(self, C_obs, G, **kwargs):
        if G is None:
            G = self.estimate_G(
                C_obs, **restrict_kwargs(self.estimate_G, kwargs))
        cphase = self._link(
            C_obs, G, **restrict_kwargs(self.link, kwargs))
        return cphase

    @abstractmethod
    def _link(self, C_obs, G):
        # C_obs, G of shape (N, P, P)
        pass

    def estimate_G(self, C_obs, **kwargs):
        G = np.abs(C_obs).real
        if self.regularizer is not None:
            G = self._regularize_G(G, **kwargs)
        else:
            G = force_doubly_nonnegative(G, inplace=True)
        return G

    def _regularize_G(self, G, **kwargs):
        return self.regularizer.regularize(G, **kwargs)

class EMILinker(Linker):
    def __init__(self, regularizer=None):
        if regularizer is not None:
            self.regularizer = regularizer
        else:
            self.regularizer = IdleRegularizer()

    def _link(self, C_obs, G):
        from greg import EMI
        return EMI(C_obs, G=G, corr=False)


class EMILinker_py(EMILinker):
    def __init__(self, regularizer=None):
        if regularizer is not None:
            self.regularizer = regularizer
        else:
            self.regularizer = IdleRegularizer()

    def _link(self, C_obs, G=None):
        from greg.linking import EMI_py as _EMI
        return _EMI(C_obs, G=G, corr=False)


class NearestNeighborLinker(Linker):
    '''
        daisy chain or cumulative product of off-diagonal elements of the covariance matrix
    '''

    def _link(self, C_obs, G=None, corr=True, **kwargs):
        nn = np.cumprod(np.diagonal(C_obs, 1))
        # append 1 to the beginning of phase history
        # nn = np.insert(nn, 0, np.exp(1j*0))
        return nn


class EVDLinker(Linker):
    '''
        Basic eigenvector decomposition of the covariance matrix
    '''

    def __init__(self, regularizer=None):
        if regularizer is not None:
            self.regularizer = regularizer
        else:
            self.regularizer = IdleRegularizer()

    def _regularize_G(self, G, **kwargs):
        return self.regularizer.regularize(G, **kwargs)

    def _link(self, C_obs, G=None):
        from greg.linking import EVD as EVD
        cphase = EVD(C_obs, G=G, corr=False)
        return cphase


if __name__ == '__main__':
    eps = 0.4
    C_obs = np.array([[3, 1j, eps + 1j], [-1j, 4, 1], [eps -1j, 1, 4]])[np.newaxis, ...]
    ceig = EMILinker().link(C_obs)
    print(ceig)

    