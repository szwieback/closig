'''
    How does velocity bias depend on mixing of subsidence signals?
    This is also a comparison of the discrete vs continuous models
'''
from closig.model import TiledCovModel, HomogSoilLayer, Geom, ContHetDispModel
from closig.expansion import SmallStepBasis, TwoHopBasis
from scripts.plotting import triangle_plot
from matplotlib import pyplot as plt
from closig.linking import EMI, EMI_py, CutOffRegularizer, NearestNeighbor, EVD
import numpy as np
P = 60
P_year = 30
dz = -0.5 * 0.05 / P_year
center = HomogSoilLayer(dz=dz/2)
trough = HomogSoilLayer(dz=dz)
fractions = [0.8, 0.2]
model_disc = TiledCovModel([center, trough], fractions=fractions)
model_cont = ContHetDispModel(
    means=[dz/2, dz], stds=[2e-4, 1e-4], weights=fractions, hist=True)


for model, name in zip([model_disc, model_cont], ['Discrete', 'Continuous']):

    error = model.covariance(P, coherence=True, displacement_phase=False) * \
        model.covariance(P, coherence=True, displacement_phase=True).conj()

    plt.scatter(model.get_baselines(P).flatten(),
                np.angle(error.flatten()), s=10, alpha=0.5)
    plt.xlabel('Baseline')
    plt.title(f'Phase error, {name}')
    plt.ylabel('Error')
    plt.show()

    taus = [2, 5, 10, 20, 30, 40, 45, 60]
    colors = plt.cm.viridis(np.linspace(0, 1, len(taus)))
    G = np.abs(model.covariance(P, coherence=True, displacement_phase=True))
    pl_true = EVD().link(model.covariance(P, coherence=True, displacement_phase=True), G=G)

    for tau, color in zip(taus, colors):
        G = np.abs(model.covariance(P, coherence=True))
        G = CutOffRegularizer().regularize(G, tau_max=tau)
        cov = model.covariance(P, coherence=True, displacement_phase=False)
        cov = G * np.exp(1j * np.angle(cov))
        pl_evd = EVD().link(cov, G=G)
        plt.plot(np.angle(pl_evd * pl_true.conj()),
                 label=f'tau: {tau}', color=color)

    plt.legend(loc='best')
    plt.xlabel('t')
    plt.ylabel('Phase [rad]')
    plt.title(f'Phase History Relative to True Phase History, {name}')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

''''
Notes:

Temporal baseline dependence is the same as in the case of the heterogenous velocity simulations I conducted earlier with a perscribed velocity field. 
Set stds to [0,0] to get exactly the same results as in the tiled model. 

With different stds, the results can behave quite wildly.

'''
