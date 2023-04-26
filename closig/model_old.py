'''
Created on Apr 25, 2023

@author: simon
'''
import numpy as np
import warnings
from closig.model import CovModel

class DiffDispTile(CovModel):
    def __init__(self, intens, dcoh, dphi, coh0=None):
        warnings.warn("DiffDispTile class deprecated")
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
        warnings.warn("VegModel class deprecated")
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

