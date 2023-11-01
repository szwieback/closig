'''
Created on Nov 1, 2023

@author: simon
'''


from abc import abstractmethod
import numpy as np

from closig import EVD, CutOffRegularizer 


class Experiment():
    @abstractmethod
    def __init__(self):
        pass
    
class CutOffExperiment(Experiment):
    def __init__(self, model, dps, P=90, P_year=30):
        self.dps = dps
        self.model = model
        self.P = P
        self.P_year = P_year
        self.Cd = model.covariance(P, coherence=True, displacement_phase=True)
        self.C = model.covariance(P, coherence=True)

    def phase_history_displacement(self):
        return EVD().link(self.Cd)

    def phase_history(self, dps=None):
        if dps is None:
            dps = self.dps
        C = self.C
        ph = np.array([EVD(CutOffRegularizer(dp)).link(C) for dp in dps])
        return ph 

    def phase_history_error(self, dps=None):
        ph = self.phase_history(dps=dps)
        phd = self.phase_history_displacement()
        sl = (None, ) * (len(ph.shape) - len(phd.shape)) + (Ellipsis, )
        return ph * phd[sl].conj() / np.abs(phd[sl])

    def phase_error(self, p0=0):
        ph = self.C[p0, :]
        phd = self.Cd[p0, :]
        phe = ph * phd.conj() / np.abs(phd)
        return np.arange(len(ph)) - p0, phe