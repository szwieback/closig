'''
Created on Nov 1, 2023

@author: simon
'''

from abc import abstractmethod
import numpy as np

from closig import EVD, CutOffRegularizer

class Experiment():
    default_L = 169
    default_seed = 12345

    @abstractmethod
    def __init__(self):
        pass

    def _L(self, L):
        looks = Experiment.default_L if L is None else L
        return looks

    def _seed(self, seed):
        _seed = Experiment.default_seed if seed is None else seed
        return _seed

    def observed_covariance(self, samples=(1024,), L=None, rng=None, seed=None):
        from greg import circular_normal, covariance_matrix
        _size = samples + (self._L(L), self.P,)
        C = self.C
        if rng is None: rng = np.random.default_rng(self._seed(seed))
        y = circular_normal(_size, C, rng=rng)
        C_obs = covariance_matrix(y)
        return C_obs

    def _correlation_matrix(self, C, corr=True):
        if not corr:
            return C
        else:
            from greg import correlation
            return correlation(C)

    def phase_history_displacement(self, corr=True):

        Cd = self._correlation_matrix(self.Cd, corr=corr)
        return EVD().link(Cd)

    def phase_history_error(self, C=None, dps=None):
        if C is None: C = self.C
        ph = self.phase_history(C=C, dps=dps)
        phd = self.phase_history_displacement()
        sl = (None,) * (len(ph.shape) - len(phd.shape)) + (Ellipsis,)
        error = ph * phd[sl].conj()
        error /= np.abs(error)
        return error
    
    def cos_metric(self, error, axis=(0,)):
        return 0.5 * (1 - np.mean(np.real(error), axis=axis))
    
    def mean_metric(self, error, axis=(0,)):
        return np.angle(np.mean(error, axis=axis))

    def phase_error(self, C=None, p0=0):
        if C is None: C = self.C
        ph = C[:,p0]
        phd = self.Cd[:,p0]
        phe = ph * phd.conj() / np.abs(phd)
        return np.arange(len(ph)) - p0, phe

    def basis_closure(self, Basis, C=None, compl=True):
        if C is None: C = self.C
        basis = Basis(P=self.P)
        closures = basis.evaluate_covariance(C, compl=compl)
        return basis, closures


class CutOffExperiment(Experiment):
    def __init__(self, model, dps, P=90, P_year=30):
        self.dps = dps
        self.model = model
        self.P = P
        self.P_year = P_year
        self._Cd = model.covariance(P, coherence=True, displacement_phase=True)
        self.C = model.covariance(P, coherence=True)

    def _dp_years(self, dps=None):
        if dps is None: dps = self.dps
        return np.array(dps) / self.P_year

    @property
    def p(self):
        return np.arange(self.P)

    @property
    def Cd(self):
        return self.Cd

    def phase_history(self, C=None, dps=None, corr=True):
        if dps is None: dps = self.dps
        if C is None: C = self.C
        _C = self._correlation_matrix(C, corr=corr)
        ph = np.stack([EVD(CutOffRegularizer(dp)).link(_C) for dp in dps], axis=-2)
        return ph
