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
    n_mean = 1.07 - 0.010j
    n_amp = 0.01 - 0.005j
    n_std = 0.001
    if scenario in ('diffdisp', 'diffdispd'):
        fractions = [0.8, 0.2]
        dz = -0.02 / P_year
        # expect phase linking error to increase with coh0 for tau >> 1
        coh0 = {'diffdisp': 0.3, 'diffdispd': 0.0}[scenario]
        dcoh = 0.6
        center = HomogSoilLayer(dz=-dz * (fractions[1] / fractions[0]), coh0=coh0, dcoh=dcoh)
        trough = HomogSoilLayer(dz=dz, coh0=coh0, dcoh=dcoh)
        model = TiledModel([center, trough], fractions=fractions, geom=geom)
    elif scenario in ('veg', 'seasonalveg', 'seasonaltrendveg', 'seasonalvegd', 'seasonaltrendvegd'):
        h = 0.4
        n_t = -0.02 / 3 if 'trend' in scenario else 0.0
        dcoh = 0.6
        if 'seasonal' not in scenario: n_amp = 0.00
        coh0 = 0.0 if scenario[-1] == 'd' else 0.3
        coh0s = 0.7
        density = 1 / h
        svl = SeasonalVegLayer(
            n_mean=n_mean, n_amp=n_amp, n_std=n_std, n_t=n_t, density=density, dcoh=dcoh, coh0=coh0, h=h,
            P_year=P_year,)
        sl = HomogSoilLayer(coh0=coh0s)
        model = LayeredCovModel([svl, sl])
    elif scenario in ('precipsoil', 'seasonalprecipsoil', 'harmonicsoil', 'seasonalsoil', 'decorrsoil'):
        if scenario == 'precipsoil':
            sl = PrecipScattSoilLayer(
                interval=10, tau=1, dcoh=0.8, coh0=0.5)
        elif scenario == 'seasonalsoil' or scenario == 'harmonicsoil':
            period = {'seasonalsoil': P_year, 'harmonicsoil': 10}
            P_max = 128
            dielModel = DobsonDielectric()
            state = 0.2 - 0.1 * np.cos(2 * np.pi * np.arange(P_max) / period[scenario])
            n = np.array([dielModel.n(s) for s in state])
            sl = ScattSoilLayer(n=n, dcoh=0.8, coh0=0.5)
        elif scenario == 'seasonalprecipsoil':
            sl = PrecipScattSoilLayer(
                interval=3, tau=1, dcoh=0.8, coh0=0.5, P_year=P_year, seasonality=0.8)
        elif scenario == 'decorrsoil':
            P_max = 128
            dielModel = DobsonDielectric()
            state = 0.2
            n = np.ones(P_max) * dielModel.n(state)
            sl = ScattSoilLayer(n=n, dcoh=0.8, coh0=0.2)
        model = LayeredCovModel([sl])
    else:
        raise ValueError(f"Scenario {scenario} not known")
    return model
