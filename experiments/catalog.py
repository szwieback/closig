'''
Created on Oct 27, 2023

@author: simon
'''
import numpy as np

def model_catalog(scenario, P_year=30):
    from closig.model import (
        LayeredCovModel, TildedModel, HomogSoilLayer, SeasonalVegLayer, Geom, PrecipScatterSoilLayer)
    geom = Geom(theta=30 * np.pi / 180, wavelength=0.24)
    h = 0.4
    n_mean = 1.05 - 0.0010j
    n_amp = 0.01 - 0.0002j
    n_std = 0.002
    if scenario == 'diffdisp':
        dz = -0.5 * geom.wavelength / P_year # half a wavelength
        coh0 = 0.6
        center = HomogSoilLayer(dz=0.00, coh0=coh0)
        trough = HomogSoilLayer(dz=dz, coh0=coh0)
        model = TildedModel([center, trough], fractions=[0.8, 0.2])
    elif scenario in ('seasonalveg', 'seasonaltrendveg'):
        n_t = {'seasonalveg': 0.0, 'seasonaltrendveg': 0.1 - 0.002j}[scenario]
        svl = SeasonalVegLayer(
            n_mean=n_mean, n_amp=n_amp, n_std=n_std, n_t=n_t, density=1/h, dcoh=0.6, P_year=P_year, h=h)
        sl = HomogSoilLayer()
        model = LayeredCovModel([svl, sl])
    elif scenario == 'precipsoil':
        model = PrecipScatterSoilLayer(
            f=10, tau=1, dcoh=0.99, coh0=0, offset=0.1, scale=0.1)
    else:
        raise ValueError(f"Scenario {scenario} not known")
    return model