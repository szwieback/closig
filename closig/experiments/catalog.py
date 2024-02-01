'''
Created on Oct 27, 2023

@author: simon
'''
import numpy as np

def model_catalog(scenario, P_year=30, band='C'):
    from closig.model import (
        LayeredCovModel, TiledModel, HomogSoilLayer, SeasonalVegLayer, Geom, PrecipScattSoilLayer, 
        ScattSoilLayer, DobsonDielectric)
    geomL = Geom(theta=30 * np.pi / 180, wavelength=0.24)
    geomC = Geom(theta=30 * np.pi / 180, wavelength=0.05)
    geom = geomC if band == 'C' else geomL
    h = 0.4
    n_mean = 1.05 - 0.0010j
    n_amp = 0.01 - 0.0002j
    n_std = 0.001
    if scenario == 'diffdisp':
        dz = -0.02 / P_year
        coh0 = 0.3 # expect phase linking error to increase with coh0 for tau >> 1
        dcoh = 0.6
        center = HomogSoilLayer(dz=0, coh0=coh0, dcoh=dcoh)
        trough = HomogSoilLayer(dz=dz, coh0=coh0, dcoh=dcoh)
        fractions = [0.8, 0.2]
        model = TiledModel([center, trough], fractions=fractions, geom=geom)
    elif scenario in ('seasonalveg', 'seasonaltrendveg'):
        n_t = {'seasonalveg': 0.0, 'seasonaltrendveg': 0.03/3}[scenario]
        # dcoh = 0.6
        svl = SeasonalVegLayer(
            n_mean=n_mean, n_amp=n_amp, n_std=n_std, n_t=n_t, density=1/h, dcoh=0.7, P_year=P_year, h=h)
        sl = HomogSoilLayer(coh0=0.6)
        model = LayeredCovModel([svl, sl])
    elif scenario in ('precipsoil', 'seasonalprecipsoil', 'harmonicsoil', 'seasonalsoil', 'decorrsoil'):
        if scenario == 'precipsoil':
            sl = PrecipScattSoilLayer(
                interval=10, tau=1, dcoh=0.8, coh0=0.1)
        elif scenario == 'seasonalsoil' or scenario == 'harmonicsoil':
            period = {'seasonalsoil': P_year, 'harmonicsoil': 10}
            P_max = 128
            dielModel = DobsonDielectric()
            state = 0.2 - 0.1 * np.cos(2 * np.pi * np.arange(P_max) / period[scenario])
            n = np.array([dielModel.n(s) for s in state])
            sl = ScattSoilLayer(n=n, dcoh=0.8, coh0=0.1)
        elif scenario == 'seasonalprecipsoil':
            sl = PrecipScattSoilLayer(
                interval=3, tau=1, dcoh=0.8, coh0=0.1, P_year=P_year, seasonality=0.8)
        elif scenario == 'decorrsoil':
            P_max = 128
            dielModel = DobsonDielectric()
            state = 0.2
            n = np.ones(P_max) * dielModel.n(state)
            sl = ScattSoilLayer(n=n, dcoh=0.8, coh0=0.1)            
        model = LayeredCovModel([sl])
    else:
        raise ValueError(f"Scenario {scenario} not known")
    return model