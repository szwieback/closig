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
        coh = coh0 + (1 - coh0) * dcoh ** (abs(p0 - p1))
    return coh

def soil_moisture_precip(
        interval, tau, unit_filter_length=20, P0=5, P_year=30, P=1024, porosity=0.35, residual=0.02, 
        seasonality=0.0):
    '''
        Generate a normalized soil moisture timeseries fed by uniformly spaced precipitation pulses
    '''
    impulses = np.zeros(P)
    precip_eff = porosity * ((1 - seasonality) + seasonality * np.cos(2*np.pi*np.arange(P)/P_year))
    impulses[P0::interval] = precip_eff[P0::interval]
    # Exponential decay
    filter_length = int(unit_filter_length / tau)
    kernel = np.exp(-1 * np.arange(0, filter_length) * tau)
    sm = np.convolve(impulses, kernel)
    sm[sm > porosity] = porosity
    sm[sm < residual] = residual
    return sm

class DielectricModel():
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def permittivity(self, state):
        pass

    def n(self, state):
        return np.sqrt(self.permittivity(state))

class DobsonDielectric(DielectricModel):
    def __init__(self, porosity=0.35, sand=0.3, clay=0.2, eps_s=4.7):
        self.sand = sand  # fraction
        self.clay = clay
        self.porosity = porosity
        self.eps_s = eps_s
        self.alpha = 0.65

    @property
    def _beta(self):
        beta_r = 1.2748 - 0.519 * self.sand - 0.152 * self.clay
        beta_i = 0.0133797 - 0.603 * self.sand - 0.166 * self.clay
        return beta_r - 1j * beta_i

    @property
    def _sigma_eff(self):
        # hard-coded bulk density
        return 1.645 + 1.939 * 1.7 - 2.013 * self.sand + 1.594 * self.clay

    @property
    def _eps_fw(self):
        eps_fw = 78 - 1j * 18  # need to implement debye
        return eps_fw

    def permittivity(self, state):
        eps_s_alpha = self.eps_s ** self.alpha
        beta = self._beta
        eps_fw_alpha = self._eps_fw ** self.alpha
        eps_alpha = 1 + (1 - self.porosity) * (eps_s_alpha - 1) + state ** beta * (eps_fw_alpha - 1) - state
        eps = eps_alpha ** (1 / self.alpha)
        return eps

class Geom():
    '''
        far field, downwelling (need to flip for backscattered waves)
    '''

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
    def __init__(self, geom=None, name=None):
        self.geom = geom
        if name is None:
            name = self.class_name
        self.name = name

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

    def covariance(self, P, coherence=False, displacement_phase=False, geom=None, subset=None):
        '''
            P: length of stack
            coherence: return normalized/coherence matrix instead of covariance matrix
            displacement_phase: return covariance matrix with displacement phase only
            geom: geometry object
            subset: subset covariance matrix with an adjacency matrix G
        '''
        C = np.zeros((P, P), dtype=np.complex64)
        for p0 in range(P):
            C[p0, p0] = self._covariance_element(p0, p0, geom=geom)
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

        if subset is not None:
            return C * subset
        else:
            return C.astype(np.complex128)

    @classmethod
    def baselines(cls, P):
        mg = np.mgrid[0:P, 0:P]
        return mg[1] - mg[0]

    @property
    def class_name(self):
        return 'covariance model'

    def __repr__(self):
        return str(self.layer_details)

    @property
    def layer_details(self):
        return self.tile_details

    @property
    def tile_details(self):
        return {
            'covmodels': [self.name],
            'fractions': [1],
        }

    def plot_diagram(self, **kwargs):
        from closig.visualization import model_diagram as md
        md.illustrate_model(self.layer_details, **kwargs)

    def plot_matrices(self, P, coherence=True, displacement_phase=False, **kwargs):
        from closig.visualization import covariance as cov_vis
        cov_vis.plot_matrices(self.covariance(
            P, coherence=coherence, displacement_phase=displacement_phase), **kwargs)

    @ property
    def _default_geom(self):
        if self.geom is not None:
            return self.geom
        else:
            return Geom()

class LayeredCovModel(CovModel):
    '''
    In-series, layered, composition of covariance models.
    Topmost layer comes first, last layer determines displacement.
    No reflection at interface
    '''

    def __init__(self, layers, geom=None, name=None):
        self.layers = layers
        self.geom = geom
        if name is None:
            name = self.class_name
        self.name = name

    def _cumulative_transmissivity(self, p, geom=None):
        if geom is None:
            geom = self._default_geom
        t = np.array([l._transmissivity(p, geom=geom) for l in self.layers])
        return np.cumprod(t)

    def _propagation_phasor(self, p0, p1, geom=None):
        if geom is None:
            geom = self._default_geom
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

    @property
    def class_name(self):
        return 'layered covariance model'

    @property
    def layer_details(self):
        return [l.tile_details for l in self.layers]

class ContHetDispModel(CovModel):
    ''''
        Represents a continuous distribution of displacements within a single model block
        This is akin to the old heterogenous velocity simulations - Rowan
    '''

    def __init__(
            self, means=[], stds=[], weights=[0.5, 0.5], L=500, dcoh=0.9, coh0=0.0,
            seed=678, name=None, geom=None):
        self.means = means
        self.stds = stds
        self.dcoh = dcoh
        self.coh0 = coh0
        self.velocities = np.zeros(0)
        self.seed = seed
        self.geom = geom
        assert np.sum(weights) == 1  # weights must sum to 1
        for mean, std, weight in zip(means, stds, weights):
            N = (int(L * weight))
            rng = np.random.default_rng(self.seed)
            self.velocities = np.concatenate((self.velocities, rng.normal(loc=mean,
                                                                          scale=std, size=N)))
        if name is None:
            name = self.class_name
        self.name = name

    def _transmissivity(self, p, geom=None):
        return 0.0

    def _displacement_phase(self, p0, p1, geom=None):
        if geom is None:
            geom = self._default_geom
        return geom.phase_from_dz(np.mean(self.velocities), p0, p1)

    def _covariance_element(self, p0, p1, geom=None):
        # multilooked phase from multilooked motions
        if geom is None:
            geom = self._default_geom
        # phases = np.exp(1j * 4 * np.pi * self.velocities *
        #                 (p1 - p0) / geom.wavelength)
        coh = coherence_model(p0, p1, self.dcoh, coh0=self.coh0)

        phases = np.exp(1j * geom.phase_from_dz(self.velocities, p0, p1)) * coh
        return np.mean(phases)

    @property
    def class_name(self):
        return 'continuously distributed displacement model'

class Layer(CovModel):

    @ abstractmethod
    def __init__(self):
        pass

    @ abstractmethod
    def _transmissivity(self, p, geom=None):
        pass

class TiledModel(CovModel):
    '''
        In-parallel, single layer, composition of covariance models.
        Represents a weighted sum of adjacent physical processes.
    '''

    def __init__(self, covmodels, fractions=None, name=None, geom=None):
        self.covmodels = covmodels
        # array of area fractions (sums to 1)
        self.fractions = fractions if fractions is not None else np.ones(
            len(covmodels)) / len(covmodels)
        assert np.all(np.isclose(np.sum(self.fractions), 1)
                      ), 'Fractions must sum to 1.'
        self.geom = geom
        if name is None:
            name = self.class_name
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

    @property
    def tile_details(self):
        return {
            'covmodels': [cov.name for cov in self.covmodels],
            'fractions': self.fractions,
        }

    @property
    def class_name(self):
        return 'tiled covariance model'

class HomogSoilLayer(Layer):
    '''
        Represents a single layer of homogenous soil with time invariant dielectric properties.
        Used for modeling displacements of the soil rather than changes of the soil.
    '''

    def __init__(self, intens=1.0, dcoh=0.9, coh0=0.0, dz=0.0, name=None):
        self.intens = intens  # array or scalar
        self.dcoh = dcoh  # coherence model: gamma_ij= | i - j | ** dcoh + coh0
        self.coh0 = coh0
        self.dz = dz
        if name is None:
            name = self.class_name
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

    @property
    def class_name(self):
        return 'homogeneous soil layer'

class ScattLayer(Layer):
    def __init__(
            self, n, density=1.0, dcoh=0.5, coh0=0.0, h=0.4, name=None):
        # refractive index array (engineering sign convention: nr - j ni)
        self.n = n
        self.density = density  # dcross section/dz; time invariant
        self.dcoh = dcoh
        self.coh0 = coh0
        self.h = h  # canopy height/layer depth [None: infinity]
        if name is None:
            name = self.class_name
        self.name = name

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
        coh = coherence_model(p0, p1, self.dcoh, coh0=self.coh0)
        dens_eff = self.density * coh
        n_p0, n_p1 = self._n(p0), self._n(p1)
        kz0, kz1 = geom.kz(n_p0), geom.kz(n_p1)
        dkzc = kz0 - kz1.conj()
        phasor_bottom = 0 if self.h is None else exp(-2j * dkzc * (-self.h))
        c01 = 1j * dens_eff / (2 * dkzc) * (1 - phasor_bottom)
        return c01

    @property
    def class_name(self):
        return 'scattering layer'

class ScattSoilLayer(ScattLayer):
    '''
        Scattering layer corresponding to changes in soil refractive index.
        Parameterized by refractive index
    '''

    def __init__(self, n, dz=0.0, density=1.0, dcoh=0.5, coh0=0.0, name=None):
        super(ScattSoilLayer, self).__init__(
            n, density=density, dcoh=dcoh, coh0=coh0, h=None)
        self.dz = dz
        if name is None:
            name = self.class_name
        self.name = name

    def _displacement_phase(self, p0, p1, geom=None):
        # absolute phase without x component
        return geom.phase_from_dz(self.dz, p0, p1)

    @property
    def class_name(self):
        return 'scattering soil layer'

class LohmanLayer(CovModel):
    '''
        From Lohman & Burgi 2023, phase closure errors are modeled as a result of a Gaussian distribution
        of sensitivity to soil moisture. For an exponential(1, 1) distribution of s, phi = arctan(Delta mv)
    '''

    def __init__(self, mv, mean=1, var=1, L=500, dcoh=0.5, coh0=0.0, seed=678, name=None):
        self.mv = mv
        self.mean = mean
        self.var = var
        self.dcoh = dcoh
        self.coh0 = coh0
        self.seed = seed
        rng = np.random.default_rng(self.seed)
        self.s = rng.exponential(scale=var, size=L)
        if name is None:
            name = self.class_name
        self.name = name

    def plot_s(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.kdeplot(self.s, bw_adjust=0.5)
        plt.xlabel('s')
        plt.ylabel('Density')
        plt.show()

    def _transmissivity(self):
        return 0

    def _covariance_element(self, p0, p1, geom=None):
        coh = coherence_model(p0, p1, self.dcoh, coh0=self.coh0)
        phases = np.exp(1j * (self.mv[p1] - self.mv[p0]) * self.s)
        return coh * np.mean(phases)

    @property
    def class_name(self):
        return 'Lohman layer'

class LohmanPrecipLayer(LohmanLayer):
    def __init__(self, interval, tau, mean=1, var=1, dcoh=0.5, coh0=0, L=500, name=None):
        mv = soil_moisture_precip(interval, tau) * 5 + 30
        super(LohmanPrecipLayer, self).__init__(
            mv, mean=mean, var=var, L=L, dcoh=dcoh, coh0=coh0, name=name)

class PrecipScattSoilLayer(ScattSoilLayer):
    '''
        Scattering layer based on repeated impulse wetting and exponential dry-down.
        interval: interval of soil_moisture_precip
        tau: time constant of exponential drying
        dz: displacement of soil surface
    '''

    def __init__(
            self, interval=10, tau=0.5, dz=0.0, density=1.0, dcoh=0.5, coh0=0.0, P_year=30, seasonality=0.0, 
            name=None):
        self.max_P = 256
        n = self._generate_n(interval, tau, P_year=P_year, seasonality=seasonality)
        super(ScattSoilLayer, self).__init__(
            n, density=density, dcoh=dcoh, coh0=coh0, h=None)
        self.dz = dz
        if name is None:
            name = self.class_name
        self.name = name

    def _generate_n(self, f, tau, P_year=30, seasonality=0.0):
        # this should be cleaned up; max_P superfluous because deterministic
        dielModel = DobsonDielectric()
        state = soil_moisture_precip(f, tau, P=self.max_P, P_year=P_year, seasonality=seasonality)
        return np.array([dielModel.n(s) for s in state])

    def _n(self, p):
        assert p < self.max_P, f"p must be less than {self.max_P}"
        return self.n[p]

    def plot_n(self, P):
        import matplotlib.pyplot as plt
        assert P < self.max_P, f"P must be less than {self.max_P}"
        plt.plot(self.n.real[:P], '-', label='real', color='black')
        plt.plot(-1 * self.n.imag[:P], '--', label='-1 * imag', color='black')
        plt.legend(loc='best')
        plt.show()

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
            dcoh=0.5, coh0=0.0, seed=678, name=None):
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
        if name is None:
            name = self.class_name
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
        return complex(n.real - 1j * n.real / 100)

    @property
    def class_name(self):
        return 'seasonal vegetation layer'

if __name__ == '__main__':
    pass
    dielM = DobsonDielectric()
    print(dielM.permittivity(0.35))
