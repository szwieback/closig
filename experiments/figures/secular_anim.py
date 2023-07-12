from closig.model import SeasonalVegLayer, HomogSoilLayer, LayeredCovModel, TiledCovModel, ContHetDispModel, ScattSoilLayer, Geom
from closig.expansion import SmallStepBasis, TwoHopBasis
from scripts.plotting import triangle_plot
from matplotlib import pyplot as plt
from closig.linking import EMI, EMI_py, CutOffRegularizer, NearestNeighbor, EVD, SeasonalRegulizer
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import colorcet as cc
from matplotlib.cm import get_cmap
import figstyle

cmap = get_cmap("cet_CET_L20")


# base_layer = HomogSoilLayer(intens=1, dz=0.005, dcoh=1)
# base_layer.plot_matrices(90, displacement_phase=False)


veg =  SeasonalVegLayer(n_mean=1.2 - 0.1j, n_std=0, n_amp=0,
                            n_t=0.1, P_year=30, density=2, dcoh=1, h=5, name='Shrubs')

# model = LayeredCovModel([veg, base_layer])
model = veg

def run(P = 90, interval = 2, model=model, wavelength = 0.056, sample_bias_at=15, suffix=''):
    # create model

    geom_Cband = Geom(wavelength=wavelength)
    k0 = 2 * geom_Cband.k0 / 1e3

    C = model.covariance(P, coherence=True, geom=geom_Cband)
    C_true = model.covariance(P, coherence=True, displacement_phase=True, geom=geom_Cband)
    pl_true = EVD().link(C_true, G=np.abs(C_true))
    disp_true = np.angle(pl_true) / k0
    G = np.abs(C)

    ## save raw errors

    baselines = model.get_baselines(P)
    errors = np.angle(C * C_true.conj()) / k0

    np.save('./experiments/figures/output/baselines', baselines)
    np.save('./experiments/figures/output/secular_dielectric_errors', errors)
    
    # create 28 adjacency matrices
    bws = np.arange(0, P + 1, interval)
    print(bws.shape)
    print(bws)

    stack_adjacency_matrices = np.zeros((len(bws), P, P))
    stack_ph = np.zeros((len(bws), P))
    colors = cmap(np.linspace(0, 1, len(bws)))
    rmse = np.zeros(len(bws))
    bias_rate = np.zeros(len(bws))

    for i in range(len(bws)):
        # compute cutoff
        stack_adjacency_matrices[i, :, :] = CutOffRegularizer().regularize(G, tau_max=bws[i])

        # Compute timeseries solution
        cov_cut = C * stack_adjacency_matrices[i, :, :]
        pl_evd = EVD().link(cov_cut, G=stack_adjacency_matrices[i, :, :])

        difference = (pl_evd * pl_true.conj())
        stack_ph[i, :] = np.unwrap(np.angle(difference)) / k0

        bias_rate[i] = 30 * stack_ph[i, sample_bias_at] / sample_bias_at ## mm / yr 

        rmse[i] = np.sqrt(np.sum(stack_ph[i, :]**2) / P)

    # Create two panel figure
    np.save('./experiments/figures/output/secular_dielectric_bias', bias_rate)
    np.save('./experiments/figures/output/bws', bws)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    im = ax[0].imshow(stack_adjacency_matrices[0, :, :], cmap='gray', interpolation=None, origin='upper', extent=[-1, P-1, -1, P-1])
    line = ax[1].plot(stack_ph[0, :], markersize=1, color=colors[0], alpha=0.8)

    ## Coherence colorbar
    divider = make_axes_locatable(ax[0])
    caxls = divider.append_axes('right', size='5%', pad=0.08)
    cbar = fig.colorbar(im, cax=caxls)
    cbar.ax.set_title(r'[$\gamma$]')

    norm = mpl.colors.Normalize(vmin=np.min(bws),vmax=np.max(bws))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    ## bw Colorbar
    divider = make_axes_locatable(ax[1])
    caxls = divider.append_axes('right', size='5%', pad=0.08)
    cbar = fig.colorbar(sm, cax=caxls)
    cbar.ax.set_title(r'[$\mathrm{bw}$]')
    
    def set_matrix_styles():
        ax[0].spines['right'].set_color(None)
        ax[0].spines['top'].set_color(None)
        tickfreq = 15
        ticks = np.arange(-0.5, P, tickfreq)
        labels = np.linspace(0, P, len(ticks)).astype(np.int16)
        ax[0].set_xticks(ticks, labels=labels)
        ax[0].set_yticks(ticks, labels=labels)

        ax[0].grid(which='minor', color='gray', linestyle='-', linewidth=0.1, alpha=0.2)
        ax[0].tick_params(which='minor', bottom=False, left=False)

        ax[0].set_ylabel(r'Reference t (acquisition)')
        ax[0].set_xlabel(r'Secondary t (acquisition)')
        ax[0].set_title('Cutoff Coherence Matrix')

    def set_ph_styles():
        # ax[1].grid()
        ax[1].set_xlabel(r'$t$ (acquisition)')
        ax[1].set_ylabel('Error [$\mathrm{mm}$]')
        ax[1].set_title('Phase History (Dielectic Only)')

    set_matrix_styles()
    set_ph_styles()

    def init():
        return [im]

    def update(frame):
        a = im.set_array(stack_adjacency_matrices[frame, :, :])
        # line.set_alpha(0.5)
        line = ax[1].plot(stack_ph[frame, :], markersize=1, color=colors[frame], alpha=0.8)

        set_matrix_styles()
        set_ph_styles()
        return [im]

    ax[0].set_aspect('equal')
    ani = FuncAnimation(fig, update, frames=np.arange(0, len(bws), 1), init_func=init, blit=False)
    ani.save(f'./experiments/figures/output/secular{suffix}.mp4', writer='imagemagick', fps=15, dpi=300, codec='h264')
    plt.show()


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    ax.plot(bws, rmse)
    ax.set_ylabel('RMSE [mm]')
    ax.set_xlabel('bw')
    ax.spines['right'].set_color(None)
    ax.spines['top'].set_color(None)
    plt.tight_layout()
    plt.savefig('./experiments/figures/output/seasonal_cutoff_rmse.png', dpi=300)
    plt.show()





if __name__ == '__main__':
    run()