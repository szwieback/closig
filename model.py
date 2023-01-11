'''
Created on Jan 5, 2023

@author: simon
'''
import numpy as np
from abc import abstractmethod

class CovModel():
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _covariance_element(self, n0, n1):
        pass

    def _coherence_element(self, n0, n1):
        num = self._covariance_element(n0, n1)
        denom = np.sqrt(self._covariance_element(n0, n0) * self._covariance_element(n1, n1))
        return num / denom

    def covariance(self, N, coherence=False):
        C = np.zeros((N, N), dtype=np.complex64)
        for n0 in range(N):
            C[n0, n0] = self._covariance_element(n0, n0)
            for n1 in range(n0 + 1, N):
                if not coherence:
                    C01 = self._covariance_element(n0, n1)
                else:
                    C01 = self._coherence_element(n0, n1)
                C[n0, n1] = C01
                C[n1, n0] = C01.conj()
        return C

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

    def _covariance_element(self, n0, n1):
        if n0 == n1:
            coh = 1
        else:
            coh = (self.dcoh) ** (abs(n1 - n0)) + self.coh0
        cont = self.intens * coh * np.exp(1j * self.dphi * (n1 - n0))
        return np.sum(cont)

class VegModel(CovModel):
    def __init__(
            self, wavelength=0.2, h=0.4, Fg=0.5, fc=1.0, dcohg=0.7, dcohc=0.5, cohig=0.8, cohic=0.2,
            ns=1.3, na=0.1, nstd=0.02, Nyear=30, seed=678):
        # Nyear: scenes per year
        self.wavelength = wavelength
        self.h = h
        self.Fg, self.fc = Fg, fc
        self.dcohg, self.dcohc = dcohg, dcohc
        self.cohig, self.cohic = cohig, cohic
        self.nm, self.na, self.nstd = ns, na, nstd
        self.Nyear = Nyear
        self.seed = seed

    def _draw_n(self, n0):
        rng = np.random.default_rng(self.seed + n0)
        ntilde = rng.normal(0, self.nstd, 1)
        n = self.nm + ntilde + self.na * np.cos(2 * np.pi * n0 / self.Nyear)
        return float(n)

    def __coherence(self, dcoh, cohi, n0, n1):
        if n0 == n1:
            return 1
        else:
            return (dcoh) ** (abs(n1 - n0)) + cohi

    def _covariance_element(self, n0, n1):
        k = 2 * np.pi / self.wavelength
        dn = self._draw_n(n0) - self._draw_n(n1)
        dn = dn * 1e-1
        phase_tot = 2 * k * dn * self.h
        if np.abs(phase_tot) < np.pi * 1e-6:
            integral = self.h
        else:
            integral = (np.exp(1j * phase_tot) - 1) / (2j * k * dn)
        cohg = self.__coherence(self.dcohg, self.cohig, n0, n1)
        cohc = self.__coherence(self.dcohc, self.cohic, n0, n1)
        C01g = self.Fg * cohg * np.exp(1j * phase_tot)
        C01c = self.fc * cohc * integral
        return C01g + C01c

if __name__ == '__main__':
    N = 91
    Nyear = 30

    dphase = 2 * np.pi / Nyear
    intensities = [0.8, 0.2]
    model = DiffDispTile(intensities, [0.30, 0.30], [0.0, dphase], coh0=[0.6, 0.6])
    h = 0.4
    model = VegModel(wavelength=0.05, h=h, Fg=0.5, fc=0.5 / h, Nyear=Nyear, nstd=0.01, ns=1.40, na=0.15)
    C = model.covariance(N)
    from expansion import TwoHopBasis, SmallStepBasis
    b = SmallStepBasis(N)
    # b = TwoHopBasis(N)
    closures = b.evaluate_covariance_matrix(C) * 180 / np.pi

    import matplotlib.pyplot as plt
    import colorcet as cc
    nt, ntau = b.nt, b.ntau
    fig, ax = plt.subplots(1, 1)
    osf = 4
    ntr = np.arange(min(nt), max(nt), step=1/osf)
    ntaur = np.arange(min(ntau), max(ntau), step=0.5/osf)

    # closures2d = np.ones((len(ntr), len(ntaur)), dtype=closures.dtype) * np.nan
    # for _jnt, _nt in enumerate(ntr):
    #     for _jntau, _ntau in enumerate(ntaur):
    #         jz = np.nonzero(np.logical_and(nt == _nt, ntau == _ntau))[0]
    #         try:
    #             closures2d[_jnt, _jntau] = closures[jz]
    #         except:
    #             pass
    from scipy.interpolate import griddata
    mg = tuple(np.meshgrid(ntr, ntaur))
    closure_grid = griddata(np.stack((nt, ntau), axis=1), closures[:, 0], mg, method='linear')
    vabs = np.nanmax(np.abs(closure_grid))
    extent = (ntaur[0] - 0.5, ntaur[-1] + 0.5, ntr[0] - 0.5, ntr[-1] + 0.5)
    xticks = np.array([15, 30, 45, 60])
    ax.set_xticks(xticks + 1)    
    ax.imshow(closure_grid[::-1,:], aspect=1, cmap=cc.cm['bjy'], vmin=-vabs, vmax=vabs, extent=extent, zorder=4)
    lw=1.0
    lc='#ffffff'
    ax.plot((extent[2] + 2, np.mean(ntr) + 1), (extent[0], ntaur[-1] - 2), lw=lw, c=lc)
    ax.plot((np.mean(ntr) + 2, extent[3] + 1), (ntaur[-1] - 2, extent[0]), lw=lw, c=lc)
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    ax.grid(True, zorder=0, axis='x', color='#eeeeee', lw=0.5)
    plt.show()

