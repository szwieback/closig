'''
Created on Jan 5, 2023

@author: simon
'''
import numpy as np
from abc import abstractmethod
from cmath import exp
from matplotlib import pyplot as plt
import closig.visualization.model_diagram as md
from closig.expansion import SmallStepBasis, TwoHopBasis
import seaborn as sns
# from closig.linking import EMI


def coherence_model(p0, p1, dcoh, coh0=0.0):
    if p0 == p1:
        coh = 1.0
    else:
        coh = (dcoh) ** (abs(p1 - p0)) + coh0
    return coh


class Geom():
    # far field, downwelling (need to flip for backscattered waves)
    def __init__(self, theta=0.0, wavelength=0.05):
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
    '''
        Parent covariance model
    '''
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

    def covariance(self, P, coherence=False, displacement_phase=False, geom=None):
        C = np.zeros((P, P), dtype=np.complex64)
        for p0 in range(P):
            C[p0, p0] = self._covariance_element(p0, p0)
            for p1 in range(p0, P):
                if not coherence:
                    C01 = self._covariance_element(p0, p1, geom=geom)
                else:
                    C01 = self._coherence_element(p0, p1, geom=geom)
                if displacement_phase:
                    C01 = np.abs(C01) * np.exp(1j *
                                               self._displacement_phase(p0, p1, geom=geom))
                C[p0, p1] = C01
                C[p1, p0] = C01.conj()

        if coherence:
            assert np.isclose(np.abs(np.diag(C)), np.ones(P)).all(
            ), 'Diagonal elements of coherence matrix should be 1'
        return C

    def link(self, P):
        '''
            extract phase history from covariance matrix
        '''
        pass

    def get_layer_details(self):
        return self.get_tile_details()

    def get_tile_details(self):
        return {
            'covmodels': [self.name],
            'fractions': [1],
        }

    def plot_diagram(self, **kwargs):
        md.illustrate_model(self.get_layer_details(), **kwargs)

    def plot_matrices(self, P, coherence=True, displacement_phase=True, ax: np.ndarray = None):
        C = self.covariance(P, coherence=coherence,
                            displacement_phase=displacement_phase)
        if ax is None:
            fig, ax = plt.subplots(1, 2)
        else:
            assert len(ax) == 2, 'Need two axes for coherence and phase.'
        ax[0].set_title('Coherence')
        ax[0].imshow(np.abs(C), vmin=0, vmax=1, cmap=plt.cm.viridis)

        ax[1].set_title('Phase')
        ax[1].imshow(np.angle(C), vmin=-np.pi, vmax=np.pi, cmap=plt.cm.seismic)
        if ax is None:
            plt.show()

    @ property
    def _default_geom(self):
        return Geom()


class LayeredCovModel(CovModel):
    '''
    In-series, layered, composition of covariance models. Topmost layer comes first, last layer determines displacement.
    No reflection at interface
    '''

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
        if geom is None:
            geom = self._default_geom
        return self.layers[-1]._displacement_phase(p0, p1, geom=geom)

    def _covariance_element(self, p0, p1, geom=None):
        if geom is None:
            geom = self._default_geom
        phasors = self._propagation_phasor(p0, p1, geom=geom)
        dphase = self._displacement_phase(p0, p1, geom=geom)
        covs_ind = np.array([l._covariance_element(
            p0, p1, geom=geom) for l in self.layers])
        cov = np.sum(covs_ind * phasors[:-1]) * exp(1j * dphase)
        return cov

    def get_layer_details(self):
        return {l.name: l.get_tile_details() for l in self.layers}


class ContHetDispModel(CovModel):
    ''''
        Represents a continous distribution of displacements within a single model block
        This is akin to my old heterogenous velocity simulations - Rowan
    '''

    def __init__(self, means=[], stds=[], weights=[0.5, 0.5], hist=False, L=500, name=''):
        self.means = means
        self.stds = stds
        self.velocities = np.zeros(0)
        assert np.sum(weights) == 1  # weights must sum to 1

        for mean, std, weight in zip(means, stds, weights):
            print(f'Mean: {mean}, std: {std}, weight: {weight}')
            N = (int(L * weight))
            print(N)
            self.velocities = np.concatenate((self.velocities, np.random.normal(loc=mean,
                                                                                scale=std, size=N)))
        if hist:
            sns.kdeplot(self.velocities * 1000)
            plt.hist(self.velocities * 1000, bins=50)
            plt.xlabel('Velocity (mm/yr)')
            plt.show()
        self.name = name

    def _transmissivity(self, p, geom=None):
        return 0.0

    def _displacement_phase(self, p0, p1, geom=None):

        if geom is None:
            geom = self._default_geom
        return self._covariance_element(p0, p1, geom=geom)

    def _covariance_element(self, p0, p1, geom=None):
        # multilooked phase from multilooked motions
        if geom is None:
            geom = self._default_geom
        phases = np.exp(1j * 4 * np.pi * self.velocities *
                        (p1 - p0) * (12/365) / geom.wavelength)

        return np.mean(phases.real) + 1j * np.mean(phases.imag)


class TiledCovModel(CovModel):
    '''
    In-parallel, single layer, composition of covariance models.
    Represents a weighted sum of adjacent physical processes.
    '''

    def __init__(self, covmodels, fractions=None, name=''):
        self.covmodels = covmodels
        # array of area fractions (sums to 1)
        self.fractions = fractions if fractions is not None else np.ones(
            len(covmodels)) / len(covmodels)
        assert np.all(np.isclose(np.sum(self.fractions), 1)
                      ), 'Fractions must sum to 1.'
        self.name = name

    def _displacement_phase(self, p0, p1, geom=None):
        if geom is None:
            geom = self._default_geom
        dphases = np.array([cm._displacement_phase(
            p0, p1, geom=geom) for cm in self.covmodels])
        return np.sum(self.fractions * dphases)

    def _covariance_element(self, p0, p1, geom=None):
        if geom is None:
            geom = self._default_geom
        covs = np.array([cm._covariance_element(p0, p1, geom=geom)
                         for cm in self.covmodels])
        cdphases = np.array(
            [exp(1j * cm._displacement_phase(p0, p1, geom=geom)) for cm in self.covmodels])
        return np.sum(self.fractions * covs * cdphases)

    def get_tile_details(self):
        return {
            'covmodels': [cov.name for cov in self.covmodels],
            'fractions': self.fractions,
        }


class Layer(CovModel):

    @ abstractmethod
    def __init__(self):
        pass

    @ abstractmethod
    def _transmissivity(self, p, geom=None):
        pass


class HomogSoilLayer(Layer):
    '''
        Represents a single layer of homogenous soil with time invariant dielctric properties. Used to for modeling displacements of the soil rather than changes of the soil.
    '''

    def __init__(self, intens=1.0, dcoh=0.9, coh0=0.0, dz=0.0, name='Homogeonous Soil'):
        self.intens = intens  # array or scalar
        self.dcoh = dcoh  # coherence model: gamma_ij= | i - j | ** dcoh + coh0
        self.coh0 = coh0
        # z displacement between adjacent acquisitions [uplift positive for (earlier)(later)*]
        # change this to a vector to account for temporally varying subsidence
        self.dz = dz
        self.name = name

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
        if geom is None:
            geom = self._default_geom
        return geom.phase_from_dz(self.dz, p0, p1)


class ScattLayer(Layer):
    def __init__(
            self, n, density=1.0, dcoh=0.5, coh0=0.0, h=0.4):
        # refractive index array (engineering sign convention: nr - j ni)
        self.n = n
        self.density = density  # dcross section/dz; time invariant
        self.dcoh = dcoh
        self.coh0 = coh0
        self.h = h  # canopy height/layer depth [None: infinity]

    def _transmissivity(self, p, geom=None):
        # amplitude; uses exp(i(-kx+wt)) sign convention
        if geom is None:
            geom = self._default_geom
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
        if geom is None:
            geom = self._default_geom
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
        if geom is None:
            geom = self._default_geom
        return geom.phase_from_dz(self.dz, p0, p1)


class SeasonalVegLayer(ScattLayer):
    '''
        Seasonally periodic scattering layer corresponding to changes in vegetation refractive index.
        The refractive index is modeled as a complex number with a mean, annual amplitude, and trend.
        The trend is modeled as a linear change in the mean refractive index per year.
        The annual amplitude is modeled as a sinusoid with a period of P_year days.
        The mean refractive index is modeled as a sinusoid with a period of P_year days.
    '''

    def __init__(
            self, n_mean=None, n_std=0.0, n_amp=0.0, n_t=0.0, P_year=30, density=1.0, h=0.4,
            dcoh=0.5, coh0=0.0, seed=678, name='Seasonal Vegetation'):
        super(SeasonalVegLayer, self).__init__(
            None, density=density, dcoh=dcoh, coh0=coh0, h=h)
        if n_mean is None:
            n_mean = 1.2 - 0.005j
        self.n_mean = n_mean  # mean refractive index (complex)
        self.n_std = n_std  # std deviation (complex)
        self.n_amp = n_amp  # annual amplitude (complex)
        self.n_t = n_t  # trend rate (change in mean per year; complex)
        self.P_year = P_year
        self.seed = seed
        self.name = name

    def _n(self, p):
        '''
            Compute refractive index at time p.
        '''

        rng = np.random.default_rng(self.seed + p)
        n_random = rng.normal(0, self.n_std.real, 1) + \
            1j * rng.normal(0, self.n_std.imag, 1)
        costerm = np.cos(2 * np.pi * p / self.P_year)
        n = self.n_mean + self.n_amp * costerm + p / self.P_year * self.n_t + n_random
        return complex(n)


if __name__ == '__main__':
    # geom = Geom(0.6, wavelength=0.2)
    # n = 1.0 - 0.001j
    # vl = ScattLayer([n, 1.2 - 0.001j, 2 - 0.02j, 1.2 - 0.02j, 3 - 0.02j, 5 - 0.04j, 1.2 - 0.001j, 3 - 0.005j],
    #                 density=0.03, dcoh=1, h=0.2)
    # t = vl._transmissivity(0, geom=geom)
    # sl = HomogSoilLayer(dcoh=0.9)
    # m = LayeredCovModel([vl, sl])
    # plt.imshow(np.abs(m.covariance(8)), vmin=0, vmax=1, cmap='viridis')
    # plt.imshow(np.angle(m.covariance(8)), vmin=-
    #            np.pi, vmax=np.pi, cmap='seismic')

    # # plot the angle of the covariance matrix with a diverging colormap alongside the magnitude with two subplots
    # fig, (ax1, ax2) = plt.subplots(1, 2)

    # im1 = ax1.imshow(np.abs(m.covariance(8, coherence=True,
    #                                      displacement_phase=True)), vmin=0, vmax=1, cmap='viridis')
    # im2 = ax2.imshow(np.angle(m.covariance(8, coherence=True,
    #                                        displacement_phase=True)), vmin=-
    #                  np.pi, vmax=np.pi, cmap='seismic')
    # plt.show()
    # create a model with a seasonal vegetation layer
    shrubs = SeasonalVegLayer(n_mean=1.2 - 0.01j, n_std=0.0001, n_amp=0.1,
                              n_t=0.5, P_year=30, density=0.01, dcoh=0.9, h=0.05, name='Shrubs')

    canopy = SeasonalVegLayer(n_mean=1.2 - 0.01j, n_std=0.0001, n_amp=0.2,
                              n_t=0.5, P_year=30, density=0.01, dcoh=0.9, h=0.5, name='Canopy')

    center = HomogSoilLayer(dz=-0.01, dcoh=1, name='Center')
    trough = HomogSoilLayer(dz=-0.02, dcoh=1, name='Trough')
    disp = TiledCovModel([center, trough], fractions=[
        0.8, 0.2], name='Heterogenous Displacement')
    layered = LayeredCovModel([shrubs, canopy])

    fig, ax = plt.subplots(nrows=1, ncols=4)

    layered.plot_diagram(ax=ax[0])
    layered.plot_matrices(
        10, coherence=True, displacement_phase=False, ax=[ax[1], ax[2]])

    # Compute closure phases for each basis
    P = 10
    smallStep = SmallStepBasis(P)
    closure_phases = smallStep.evaluate_covariance(layered.covariance(P))
    from scripts.plotting import triangle_plot
    triangle_plot(smallStep, closure_phases, ax=ax[3])

    plt.show()

    # plt.imshow(np.abs(pm.covariance(2)), vmin=0, vmax=1, cmap='viridis')
    # plt.show()
