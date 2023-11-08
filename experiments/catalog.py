'''
Created on Oct 27, 2023

@author: simon
'''
import numpy as np

def model_catalog(scenario, P_year=30, band='C'):
    from closig.model import (
        LayeredCovModel, TiledModel, HomogSoilLayer, SeasonalVegLayer, Geom, PrecipScatterSoilLayer)
    geomL = Geom(theta=30 * np.pi / 180, wavelength=0.24)
    geomC = Geom(theta=30 * np.pi / 180, wavelength=0.05)
    geom = geomC if band == 'C' else geomL
    h = 0.4
    n_mean = 1.05 - 0.0010j
    n_amp = 0.01 - 0.0002j
    n_std = 0.002
    if scenario == 'diffdisp':
        dz = -0.025 / P_year
        coh0 = 0.3 # expect phase linking error to increase with coh0 for tau >> 1
        dcoh = 0.6
        center = HomogSoilLayer(dz=0, coh0=coh0, dcoh=dcoh)
        trough = HomogSoilLayer(dz=dz, coh0=coh0, dcoh=dcoh)
        fractions = [0.8, 0.2]
        model = TiledModel([center, trough], fractions=fractions, geom=geom)
    elif scenario in ('seasonalveg', 'seasonaltrendveg'):
        n_t = {'seasonalveg': 0.0, 'seasonaltrendveg': 0.01}[scenario]
        svl = SeasonalVegLayer(
            n_mean=n_mean, n_amp=n_amp, n_std=n_std, n_t=n_t, density=1/h, dcoh=0.6, P_year=P_year, h=h)
        sl = HomogSoilLayer()
        model = LayeredCovModel([svl, sl])
    elif scenario == 'precipsoil':
        psl = PrecipScatterSoilLayer(
            interval=10, tau=1, dcoh=0.8, coh0=0.1)
        model = LayeredCovModel([psl])
    else:
        raise ValueError(f"Scenario {scenario} not known")
    return model