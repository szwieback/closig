'''
Created on Jan 5, 2023

@author: simon
'''
import numpy as np
from abc import abstractmethod
from cmath import exp

def coherence_model(p0, p1, dcoh, coh0=0.0):
    if p0 == p1:
        coh = 1.0
    else:
        coh = (dcoh) ** (abs(p1 - p0)) + coh0
    return coh

class Geom():
    # far field, downwelling (need to flip for backscattered waves)
    def __init__(self, theta=0.0, wavelength=0.2):
        self.theta = theta  # incidence angle [rad]
        self.wavelength = wavelength  # in vacuum

    def kz(self, n=1):
        return -np.sqrt((n * self.k0) ** 2 - self.kx(n) ** 2)

    def kx(self, n=1):
        return self.k0 * np.sin(self.theta)

    @property
    def k0(self):  # scalar wavenumber in vacuum
        return 2 * np.pi / self.wavelength

    def phase_from_dz(self, dz, p0, p1):
        return -2 * self.kz() * dz * (p0 - p1)

class CovModel():
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    # s0 * conj(s1)
    def _covariance_element(self, p0, p1, geom=None):
        pass

    @abstractmethod
    def _displacement_phase(self, p0, p1, geom=None):
        pass

    def _coherence_element(self, p0, p1, geom=None):
        num = self._covariance_element(p0, p1, geom=geom)
        denom = np.sqrt(
            self._covariance_element(p0, p0, geom=geom) * self._covariance_element(p1, p1, geom=geom))
        return num / denom

    def covariance(self, N, coherence=False, displacement_phase=False, geom=None):
        C = np.zeros((N, N), dtype=np.complex64)
        for p0 in range(N):
            C[p0, p0] = self._covariance_element(p0, p0)
            for p1 in range(p0 + 1, N):
                if not coherence:
                    C01 = self._covariance_element(p0, p1, geom=geom)
                else:
                    C01 = self._coherence_element(p0, p1, geom=geom)
                if displacement_phase:
                    C01 = np.abs(C01) * np.exp(1j * self._displacement_phase(p0, p1, geom=geom))
                C[p0, p1] = C01
                C[p1, p0] = C01.conj()
        return C

    @property
    def _default_geom(self):
        return Geom()

class LayeredCovModel(CovModel):
    # works in series, topmost layer comes first; last layer determines displacement
    # no reflection at interface
    def __init__(self, layers):
        self.layers = layers

    def _cumulative_transmissivity(self, p, geom=None):
        t = np.array([l._transmissivity(p, geom=geom) for l in self.layers])
        return np.cumprod(t)

    def _propagation_phasor(self, p0, p1, geom=None):
        t0 = self._cumulative_transmissivity(p0, geom=geom)
        t1 = self._cumulative_transmissivity(p1, geom=geom)
        tc = t0 * t1.conj()
        return np.concatenate((np.array([1.0]), tc))

    def _displacement_phase(self, p0, p1, geom=None):
        if geom is None: geom = self._default_geom
        return self.layers[-1]._displacement_phase(p0, p1, geom=geom)

    def _covariance_element(self, p0, p1, geom=None):
        if geom is None: geom = self._default_geom
        phasors = self._propagation_phasor(p0, p1, geom=geom)
        dphase = self._displacement_phase(p0, p1, geom=geom)
        covs_ind = np.array([l._covariance_element(p0, p1, geom=geom) for l in self.layers])
        cov = np.sum(covs_ind * phasors[:-1]) * exp(1j * dphase)
        return cov

class TiledCovModel(CovModel):
    # works in parallel
    def __init__(self, covmodels, fractions=None):
        self.covmodels = covmodels
        # array of area fractions (sums to 1)
        self.fractions = fractions if fractions is not None else np.ones(len(covmodels)) / len(covmodels)
        assert np.all(np.isclose(np.sum(self.fractions), 1))

    def _displacement_phase(self, p0, p1, geom=None):
        if geom is None: geom = self._default_geom
        dphases = np.array([cm._displacement_phase(p0, p1, geom=geom) for cm in self.covmodels])
        return np.sum(self.fractions * dphases)

    def _covariance_element(self, p0, p1, geom=None):
        if geom is None: geom = self._default_geom
        covs = np.array([cm._covariance_element(p0, p1, geom=geom) for cm in self.covmodels])
        cdphases = np.array([exp(1j * cm._displacement_phase(p0, p1, geom=geom)) for cm in self.covmodels])
        return np.sum(self.fractions * covs * cdphases)

class Layer(CovModel):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _transmissivity(self, p, geom=None):
        pass

class HomogSoilLayer(Layer):

    def __init__(self, intens=1.0, dcoh=0.9, coh0=0.0, dz=0.0):
        self.intens = intens  # array or scalar
        self.dcoh = dcoh  # coherence model: gamma_ij= | i - j | ** dcoh + coh0
        self.coh0 = coh0
        self.dz = dz  # z displacement between adjacent acquisitions [uplift positive for (earlier)(later)*]

    def _transmissivity(self, p, geom=None):
        return 0.0

    def _intensity(self, p):
        if np.ndim(self.intens) == 0:
            return self.intens
        else:
            return self.intens[p]

    def _covariance_element(self, p0, p1, geom=None):
        # does not account for displacement because phase reference is at top of layer
        coh = coherence_model(p0, p1, self.dcoh, coh0=self.coh0)
        C01 = np.sqrt(self._intensity(p0) * self._intensity(p1)) * coh
        return C01

    def _displacement_phase(self, p0, p1, geom=None):
        # absolute phase without x component
        if geom is None: geom = self._default_geom
        return geom.phase_from_dz(self.dz, p0, p1)

class ScattLayer(Layer):
    def __init__(
        self, n, density=1.0, dcoh=0.5, coh0=0.0, h=0.4):
        self.n = n  # refractive index array (engineering sign convention: nr - j ni)
        self.density = density  # dcross section/dz; time invariant
        self.dcoh = dcoh
        self.coh0 = coh0
        self.h = h  # canopy height/layer depth [None: infinity]

    def _transmissivity(self, p, geom=None):
        # amplitude; uses exp(i(-kx+wt)) sign convention
        if geom is None: geom = self._default_geom
        if self.h is None or self.h <= 0:
            return 0.0
        else:
            n_p = self._n(p)
            return np.exp(-1j * 2 * (-self.h) * geom.kz(n_p))

    def _displacement_phase(self, p0, p1, geom=None):
        # not meaningful for vegetation canopy
        return 0.0

    def _n(self, p):
        return self.n[p]

    def _covariance_element(self, p0, p1, geom=None):
        if geom is None: geom = self._default_geom
        coh = coherence_model(p0, p1, self.dcoh, coh0=self.coh0)
        dens_eff = self.density * coh
        n_p0, n_p1 = self._n(p0), self._n(p1)
        kz0, kz1 = geom.kz(n_p0), geom.kz(n_p1)
        dkzc = kz0 - kz1.conj()
        phasor_bottom = 0 if self.h is None else exp(-2j * dkzc * (-self.h))
        c01 = 1j * dens_eff / (2 * dkzc) * (1 - phasor_bottom)
        return c01

class ScattSoilLayer(ScattLayer):

    def __init__(self, n, dz=0.0, density=1.0, dcoh=0.5, coh0=0.0):
        super(ScattSoilLayer, self).__init__(
            n, density=density, dcoh=dcoh, coh0=coh0, h=None)
        self.dz = dz

    def _displacement_phase(self, p0, p1, geom=None):
        # absolute phase without x component
        if geom is None: geom = self._default_geom
        return geom.phase_from_dz(self.dz, p0, p1)

class SeasonalVegModel(ScattLayer):
    def __init__(
        self, n_mean=None, n_std=0.0, n_amp=0.0, n_t=0.0, P_year=30, density=1.0, h=0.4,
        dcoh=0.5, coh0=0.0, seed=678):
        super(SeasonalVegModel, self).__init__(
            None, density=density, dcoh=dcoh, coh0=coh0, h=h)
        if n_mean is None: n_mean=1.2 - 0.005j
        self.n_mean = n_mean  # mean refractive index (complex)
        self.n_std = n_std  # std deviation (complex)
        self.n_amp = n_amp  # annual amplitude (complex)
        self.n_t = n_t  # trend rate (change in mean per year; complex)
        self.P_year = P_year
        self.seed = seed

    def _n(self, p):
        rng = np.random.default_rng(self.seed + p)
        n_random = rng.normal(0, self.n_std.real, 1) + 1j * rng.normal(0, self.n_std.imag, 1)
        costerm = np.cos(2 * np.pi * p / self.P_year)
        n = self.n_mean + self.n_amp * costerm + p / self.P_year * self.n_t + n_random
        return complex(n)

if __name__ == '__main__':
    geom = Geom(0.6, wavelength=0.2)
    n = 1.0 - 0.001j
    vl = ScattLayer([n, 1.2 - 0.001j, n], density=0.03, dcoh=1.0, h=0.1)
    t = vl._transmissivity(0, geom=geom)
    sl = HomogSoilLayer(dcoh=0.9)
    m = LayeredCovModel([vl, sl])
    print(np.angle(m._covariance_element(0, 1, geom=geom)))

    center = HomogSoilLayer(dz=0.00)
    trough = HomogSoilLayer(dz=-0.05)
    pm = TiledCovModel([center, trough], fractions=[0.8, 0.2])


    svl = SeasonalVegModel(n_amp=0.0500-0.0005j, P_year=10)
    svm = LayeredCovModel([svl, sl])
    print(svl._n(0), svl._n(1))
    print(np.angle(svm.covariance(5, geom=geom)[0, 1]))
    
    sl = ScattSoilLayer([2.0 - 1.0j, 2.2 - 1.2j])

