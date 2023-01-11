'''
Created on Jan 4, 2023

@author: simon
'''

import numpy as np
from abc import ABC, abstractmethod

# Covariance:
# ..., N, N

class Basis():
    def __init__(self, N):
        self.N = N
        self._initalize()

    @abstractmethod
    def _initialize(self):
        pass

    def _std_basis(self, n0, n1):
        ind = (n0 + 1) * self.N - n0 * (n0 + 1) // 2 + (n1 - n0 - self.N - 1)
        return ind

    # get covector in standard basis
    def to_std_basis(self, covector):
        xi = np.zeros(self.N * (self.N - 1) // 2, dtype=np.int16)
        for edge in covector:
            xi[self._std_basis(edge[0], edge[1])] = edge[2]
        return xi

    def basis_matrix_std_basis(self):
        return np.stack([self.to_std_basis(covector) for covector in self.covectors], axis=0)
    
    @staticmethod
    def _evaluate_covariance_matrix(C, covector, normalize=False, compl=False):
        if len(C.shape) < 2: raise ValueError(f"C of shape {C.shape}")
        _C = C if len(C.shape) > 2 else C[np.newaxis, :]
        gcoh = np.ones(_C.shape[:-2], dtype=np.complex)
        for edge in covector:
            g = _C[..., edge[0], edge[1]]
            if normalize:
                g /= np.sqrt(_C[..., edge[0], edge[0]] - _C[..., edge[1], edge[1]])
            if edge[2] == -1:
                g = g.conj()
            if np.abs(edge[2]) != 1:
                raise ValueError("Only pure cycles supported")
            gcoh *= g
        if not compl:
            gcoh = np.angle(gcoh)
        return gcoh
    
    def evaluate_covariance_matrix(self, C, covector=None, normalize=False, compl=False):
        if covector is not None:
            return self._evaluate_covariance_matrix(C, covector=covector, normalize=normalize, compl=compl)
        else:
            gcohs = []
            for covector in self.covectors:
                gcoh =  self._evaluate_covariance_matrix(
                    C, covector=covector, normalize=normalize, compl=compl)
                gcohs.append(gcoh)
            return np.array(gcohs)
    
    def _test_number(self):
        assert len(self.covectors) == (self.N - 1) * (self.N - 2) / 2

    def _test_dimension(self):
        from scipy.linalg import lu
        A = self.basis_matrix_std_basis().T
        P, L, U = lu(A)
        assert np.min(np.abs(np.diag(U))) > 1e-16

class SmallStepBasis(Basis):
    def __init__(self, N):
        self.N = N
        self._initialize()

    def _initialize(self):
        intervals, covectors = [], []
        nt, ntau = [], []
        # assembly
        for nb in range(self.N - 2):
            for ne in range(nb + 2, self.N):
                intervals.append((nb, ne))
                covectors.append(self._covector(nb, ne))
                nt.append(self._nt(nb, ne))
                ntau.append(self._ntau(nb, ne))
        # sorting
        lsort = lambda x: [x[ind] for ind in np.lexsort((nt, ntau))]
        self.ntau = np.array(lsort(ntau))
        self.nt = np.array(lsort(nt))
        self.covectors = lsort(covectors)
        self.intervals = np.array(lsort(intervals))

    @staticmethod
    def _nt(nb, ne):
        return 0.5 * (nb + ne)

    @staticmethod
    def _ntau(nb, ne):
        return ne - nb

    @staticmethod
    def _covector(nb, ne):
        return [(n, n + 1, 1) for n in range(nb, ne)] + [(nb, ne, -1)]

# class for two hop
# loop over t, tau

class TwoHopBasis(Basis):
    def __init__(self, N):
        self.N = N
        self._initialize()

    def _initialize(self):
        self.covectors = []
        intervals, nt, ntau = [], [], []
        # assembly
        for _ntau in range(2, self.N):
            hminus, hplus = self._h(_ntau)
            for _ntraw in range(hminus, self.N - hplus):
                intervals.append((_ntraw - hminus, _ntraw + hplus))
                self.covectors.append(self._covector(_ntraw, _ntau))
                _nt = _ntraw + 0.5 * (hplus - hminus)
                nt.append(_nt)
                ntau.append(_ntau)
        self.ntau, self.nt = np.array(ntau), np.array(nt)
        self.intervals = np.array(intervals)

    @staticmethod
    def _h(ntau):
        from math import floor, ceil
        return floor(ntau / 2), ceil(ntau / 2)
        
    @staticmethod
    def _covector(nt, ntau):
        hminus, hplus = TwoHopBasis._h(ntau)
        return [(nt-hminus, nt, 1), (nt, nt+hplus, 1), (nt-hminus, nt+hplus, -1)]

if __name__ == '__main__':
    L = 40
    N = 4
    S = 20
    # y = np.random.normal(size=(S, L, N)) + 1j * np.random.normal(size=(S, L, N))
    # C = np.mean(y[..., np.newaxis] * y[..., np.newaxis,:].conj(), axis=-3)
    # print(C.shape)
    # ss_covectors(N)
    for N in range(3, 5):
        b = TwoHopBasis(N)
        print(N, len(b.covectors))
        b._test_dimension()
        b._test_number()
    print(b.covectors)