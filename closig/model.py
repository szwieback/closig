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

class CovModel():
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    # s0 * conj(s1)
    def _covariance_element(self, n0, n1, displacement_phase=False):
        pass

    def _coherence_element(self, n0, n1, displacement_phase=False):
        num = self._covariance_element(n0, n1, displacement_phase=displacement_phase)
        denom = np.sqrt(self._covariance_element(n0, n0) * self._covariance_element(n1, n1))
        return num / denom

    def covariance(self, N, coherence=False, displacement_phase=False):
        C = np.zeros((N, N), dtype=np.complex64)
        for n0 in range(N):
            C[n0, n0] = self._covariance_element(n0, n0)
            for n1 in range(n0 + 1, N):
                if not coherence:
                    C01 = self._covariance_element(n0, n1, displacement_phase=displacement_phase)
                else:
                    C01 = self._coherence_element(n0, n1, displacement_phase=displacement_phase)
                C[n0, n1] = C01
                C[n1, n0] = C01.conj()
        return C

class Geom():
    # far field, downwelling
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

class Layer():
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _covariance_element(self, p0, p1, geom=None):
        # s0 * conj(s1)
        pass

    @abstractmethod
    def _transmissivity(self, p, geom=None):
        pass

    @abstractmethod
    def _displacement_phase(self, p, geom=None):
        pass

    @property
    def _default_geom(self):
        return Geom()

class HomogSoilLayer(Layer):

    def __init__(self, intens=1.0, dcoh=0.5, coh0=0.0, dz=0.0):
        self.intens = intens  # constant
        self.dcoh = dcoh  # coherence model: gamma_ij= | i - j | ** dcoh + coh0
        self.coh0 = coh0
        self.dz = dz  # z displacement between adjacent acquisitions [z0 - z1: subsidence positive]

    def _transmissivity(self, p, geom=None):
        return 0.0

    def _covariance_element(self, p0, p1, geom=None):
        # does not account for displacement
        coh = coherence_model(p0, p1, self.dcoh, coh0=self.dcoh0)
        C01 = self.intens * coh
        return C01

    def _displacement_phase(self, p0, p1, geom=None):
        # absolute phase without x component
        if geom is None: geom = self._default_geom
        return geom.phase_from_dz(self.dz, p0, p1)

class ScattLayer():
    def __init__(
        self, n, intens_density=1.0, dcoh=0.5, coh0=0.0, h=0.4):
        self.n = n  # refractive index array (check sign convention)
        self.intens_density = intens_density  # dbackscatter intensity/dz; time invariant
        self.dcoh = dcoh
        self.coh0 = coh0
        self.h = h  # canopy height/layer depth [None: infinity]

    def _transmissivity(self, p, geom=None):
        # amplitude; uses exp(i(-kx+wt)) sign convention
        if geom is None: geom = self._default_geom
        if self.h <= 0:
            return 0.0
        else:
            return np.exp(-1j * 2 * (-self.h) * geom.kz(self.n[p]))

    def _displacement_phase(self, p0, p1, geom=None):
        # not meaningful for vegetation canopy
        return 0.0

    def _covariance_element(self, p0, p1, geom=None):
        if geom is None: geom = self._default_geom
        coh = coherence_model(p0, p1, self.dcoh, coh0=self.coh0)
        dens_eff = self.intens_density * coh
        kz0, kz1 = geom.kz(self.n[p0]), geom.kz(self.n[p1])
        dkzc = kz0 - kz1.conj()
        phasor_bottom = 0 if self.h is None else exp(-2j * dkzc *(-self.h))
        c01 = 1j * dens_eff / (2 * dkzc) * (1 - phasor_bottom)
        return c01

class ScattSoilLayer(ScattLayer):
    
    def __init__(self, n, dz=0.0, intens_density=1.0, dcoh=0.5, coh0=0.0):
        super(ScattSoilLayer, self).__init__(
            n, intens_density=intens_density, dcoh=dcoh, coh0=coh0, h=None)
        self.dz = dz
        
    def _displacement_phase(self, p0, p1, geom=None):
        # absolute phase without x component
        if geom is None: geom = self._default_geom
        return geom.phase_from_dz(self.dz, p0, p1)
    

class LayeredCovModel(CovModel):
    # works in series
    def __init__(self, lst_layers):
        self.lst_layers = lst_layers

    def _cumulative_transmissivity(self, p, geom=None):
        t = np.array([l._transmissivity(p, geom=geom) for l in self.lst_layers])
        return np.cumprod(t)
        
    def _propagation_phasor(self, p0, p1, geom=None):
        t0 = self._cumulative_transmissivity(p0, geom=geom)
        t1 = self._cumulative_transmissivity(p1, geom=geom)
        tc = t0 * t1.conj()
        return tc
        
    def _covariance_element(self, p0, p1, geom=None):
        if geom is None: geom = self._default_geom
        phasors = self._propagation_phasor(p0, p1, geom=geom)
        dphase = self.lst_layers[-1]._displacement_phase(p0, p1, geom=geom)
        covs_ind = np.array([l._covariance_element(p0, p1, geom=geom) for l in self.lst_layers])
        cov = np.sum(covs_ind * dphase) * exp(1j * dphase)
        
        

class DiffDispTile(CovModel):
    def __init__(self, intens, dcoh, dphi, coh0=None):
        if len(dcoh) != len(intens):
            raise ValueError(f"Expected dcoh of length {len(intens)}")
        if len(dphi) != len(intens):
            raise ValueError(f"Expected dphi of length {len(intens)}")
        self.intens = np.array(intens)
        self.dcoh = np.array(dcoh)
        self.dphi = np.array(dphi)
        if coh0 is None:
            coh0 = np.zeros_like(self.dcoh)
        self.coh0 = coh0

    def _covariance_element(self, n0, n1, displacement_phase=False):
        if n0 == n1:
            coh = 1
        else:
            coh = (self.dcoh) ** (abs(n1 - n0)) + self.coh0
        if not displacement_phase:
            cont = self.intens * coh * np.exp(1j * self.dphi * (n1 - n0))
            C01 = np.sum(cont)
        else:
            phase = np.sum(self.intens * self.dphi * (n1 - n0)) / np.sum(self.intens)
            C01 = np.sum(self.intens) * np.exp(1j * phase)
        return C01

class VegModel(CovModel):
    def __init__(
            self, wavelength=0.2, h=0.4, Fg=0.5, fc=1.0, dcohg=0.7, dcohc=0.5, cohig=0.8, cohic=0.2,
            nrm=1.30, nra=0.10, nrstd=0.02, nrt=0.10, Nyear=30, seed=678, theta=0.0):
        # Nyear: scenes per year
        self.wavelength = wavelength
        self.h = h
        self.theta = theta
        self.Fg, self.fc = Fg, fc
        self.dcohg, self.dcohc = dcohg, dcohc
        self.cohig, self.cohic = cohig, cohic
        self.nrm, self.nra, self.nrstd, self.nrt = nrm, nra, nrstd, nrt
        self.Nyear = Nyear
        self.seed = seed

    def _draw_nr(self, n0):
        rng = np.random.default_rng(self.seed + n0)
        nrtilde = rng.normal(0, self.nrstd, 1)
        costerm = np.cos(2 * np.pi * n0 / self.Nyear)
        nr = self.nrm + self.nra * costerm + n0 / self.Nyear * self.nrt + nrtilde
        return float(nr)

    def __coherence(self, dcoh, cohi, n0, n1):
        if n0 == n1:
            return 1
        else:
            return (dcoh) ** (abs(n1 - n0)) + cohi

    def _covariance_element(self, n0, n1, displacement_phase=False):
        k = 2 * np.pi / self.wavelength
        dn = self._draw_nr(n0) - self._draw_nr(n1)
        heff = self.h / np.cos(self.theta)
        phase_tot = 2 * k * dn * heff
        if np.abs(phase_tot) < np.pi * 1e-6:
            integral = heff
        else:
            integral = (np.exp(1j * phase_tot) - 1) / (2j * k * dn)
        cohg = self.__coherence(self.dcohg, self.cohig, n0, n1)
        cohc = self.__coherence(self.dcohc, self.cohic, n0, n1)
        C01g = self.Fg * cohg * np.exp(1j * phase_tot)
        C01c = self.fc * cohc * integral
        C01 = C01g + C01c
        if displacement_phase: C01 = np.abs(C01)  # nothing moves
        return C01

if __name__ == '__main__':
    geom = Geom(0.6, wavelength=0.2)
    n = 1.0 - 0.04j
    sl = ScattLayer(np.full(2, n), dcoh=1.0, h=0.2)
    t = sl._transmissivity(0, geom=geom)
    # test layered model (compare phase when h->0 or n->1)
    # implement tile model
