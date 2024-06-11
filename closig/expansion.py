'''
Created on Jan 4, 2023

@author: simon
'''

import numpy as np
from abc import abstractmethod

# Covariance:
# ..., P, P

class Basis():
    def __init__(self, P):
        self.P = P
        self._initalize()

    @abstractmethod
    def _initialize(self):
        pass

    def _std_basis(self, p0, p1):
        if max(p0, p1) >= self.P or min(p0, p1) < 0 or p0 >= p1:
            raise ValueError(f"{p0}, {p1} not valid")
        ind = p0 * self.P - p0 * (p0 + 1) // 2 + (p1 - p0 - 1)
        return ind

    def _std_basis_vec(self, p0, p1):
        # includes self-connected edges (for "intensities")
        if max(p0, p1) >= self.P or min(p0, p1) < 0 or p0 > p1:
            raise ValueError(f"{p0}, {p1} not valid")
        ind = p0 * self.P - p0 * (p0 - 1) // 2 + (p1 - p0)
        return ind

    # get covector in standard basis
    def to_std_basis(self, covector):
        xi = np.zeros(self.P * (self.P - 1) // 2, dtype=np.int16)
        for edge in covector:
            xi[self._std_basis(edge[0], edge[1])] = edge[2]
        return xi

    def basis_matrix_std_basis(self):
        return np.stack([self.to_std_basis(covector) for covector in self.covectors], axis=0)

    def __evaluate(self, C, covector, normalize=False, vectorized=False, forward_only=False):
        _edgecombs = [(0, 1), (0, 0), (1, 1)]
        shape = C.shape[:-2] if not vectorized else C.shape[:-1]
        gcoh = np.ones(shape, dtype=np.complex256)
        for edge in covector:
            if not vectorized:
                g = C[..., edge[0], edge[1]]
                if normalize:
                    g /= np.sqrt(C[..., edge[0], edge[0]]
                                 * C[..., edge[1], edge[1]])
            else:
                inds = [self._std_basis_vec(edge[e0], edge[e1])
                        for e0, e1 in _edgecombs]
                g = C[..., inds[0]]
                if normalize:
                    # np.sqrt(C[..., inds[1]] * C[..., inds[2]])
                    g /= np.abs(g)
                    import warnings
                    warnings.warn('Normalization not based on coherence')
            if edge[2] == -1:
                g = g.conj()
                if forward_only:
                    g = np.ones_like(g)
            if np.abs(edge[2]) != 1:
                raise ValueError("Only pure cycles supported")
            gcoh *= g
        return gcoh.astype(np.complex64)

    def _evaluate_covariance(
            self, C, covector, normalize=False, compl=False, vectorized=False, forward_only=False):
        if not vectorized:
            if len(C.shape) < 2:
                raise ValueError(f"C of shape {C.shape}")
            _C = C if len(C.shape) > 2 else C[np.newaxis, :]
        else:
            _C = C if len(C.shape) > 1 else C[np.newxis, :]
        gcoh = self.__evaluate(
            C, covector, vectorized=vectorized, normalize=normalize, forward_only=forward_only)
        if not compl:
            gcoh = np.angle(gcoh)
        return gcoh

    def evaluate_covariance(
            self, C, covector=None, normalize=False, compl=False, vectorized=False, forward_only=False):
        def _eval(covector):
            gcoh = self._evaluate_covariance(
                C, covector=covector, normalize=normalize, compl=compl, vectorized=vectorized,
                forward_only=forward_only)
            return gcoh
        if covector is not None:
            return _eval(covector)
        else:
            gcohs = np.array([_eval(covector) for covector in self.covectors])
            return gcohs

    def _test_number(self):
        assert len(self.covectors) == (self.P - 1) * (self.P - 2) / 2

    def _test_dimension(self):
        from scipy.linalg import lu
        A = self.basis_matrix_std_basis().T
        P, L, U = lu(A)
        assert np.min(np.abs(np.diag(U))) > 1e-16


class SmallStepBasis(Basis):
    def __init__(self, P):
        self.P = P
        self._initialize()

    def _initialize(self):
        intervals, covectors = [], []
        pt, ptau = [], []
        # assembly
        for pb in range(self.P - 2):
            for pe in range(pb + 2, self.P):
                intervals.append((pb, pe))
                covectors.append(self._covector(pb, pe))
                pt.append(self._pt(pb, pe))
                ptau.append(self._ptau(pb, pe))
        # sorting
        def lsort(x): return [x[ind] for ind in np.lexsort((pt, ptau))]
        self.ptau = np.array(lsort(ptau))
        self.pt = np.array(lsort(pt))
        self.covectors = lsort(covectors)
        self.intervals = np.array(lsort(intervals))

    @staticmethod
    def _pt(pb, pe):
        return 0.5 * (pb + pe)

    @staticmethod
    def _ptau(pb, pe):
        return pe - pb

    @staticmethod
    def _covector(pb, pe):
        return [(p, p + 1, 1) for p in range(pb, pe)] + [(pb, pe, -1)]


class TwoHopBasis(Basis):
    def __init__(self, P):
        self.P = P
        self._initialize()

    def _initialize(self):
        self.covectors = []
        intervals, pt, ptau = [], [], []
        # assembly
        for _ptau in range(2, self.P):
            hminus, hplus = self._h(_ptau)
            for _praw in range(hminus, self.P - hplus):
                intervals.append((_praw - hminus, _praw + hplus))
                self.covectors.append(self._covector(_praw, _ptau))
                _pt = _praw + 0.5 * (hplus - hminus)
                pt.append(_pt)
                ptau.append(_ptau)
        self.ptau, self.pt = np.array(ptau), np.array(pt)
        self.intervals = np.array(intervals)

    def basis_indices(self, cutoff=None):
        '''
            Return a triple of indices for the basis vectors,
            useful in combination with intensity triplet code.
            Only valid for the two-hop expansion since each covector is restricted to three edges
        '''
        indices = np.stack([[covector[0][0], covector[0][1], covector[1][1]]
                            for covector in self.covectors], axis=0)
        if cutoff is not None:
            indices = [triplet for triplet in indices if np.max(
                triplet) < cutoff]
        return indices

    @staticmethod
    def _h(ptau):
        from math import floor, ceil
        return floor(ptau / 2), ceil(ptau / 2)

    @staticmethod
    def _covector(pt, ptau):
        hminus, hplus = TwoHopBasis._h(ptau)
        return [(pt - hminus, pt, 1), (pt, pt + hplus, 1), (pt - hminus, pt + hplus, -1)]

def clip_cclosures(p_range, basis, cclosures, axis=0):
    # returns basis and cclosures array clipped to p_range of acquisitions
    # axis: axis of cclosures spanning elements of closure basis
    if p_range[0] > basis.P or p_range[1] > basis.P or p_range[0] <=0:
        raise ValueError("Invalid p_range")
    P_clip = p_range[1] - p_range[0]
    basis_clip = type(basis)(P_clip)
    ind_elements = []
    for interval_clip in basis_clip.intervals:
        interval = interval_clip + p_range[0]
        matches = np.nonzero(
            np.logical_and(basis.intervals[:, 0] == interval[0], basis.intervals[:, 1] == interval[1]))[0]
        assert len(matches == 1)
        ind_elements.append(matches[0])
    sl = tuple([slice(dim) if jdim != axis else ind_elements for jdim, dim in enumerate(cclosures.shape)])
    cclosures_clip = cclosures[sl]
    return basis_clip, cclosures_clip
    
