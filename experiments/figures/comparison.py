import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
import figstyle

fontsize = 13
colors = ['#647381', 'black']

scenarios = ['het_vel', 'secular_dielectric']
scenarioNames = ['Heterogeneous Velocity', 'Secular Dielectric']
base_path = './experiments/figures/output'


def plot_rawerror(wavelength=0.056):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
    scatter_styles = ['x', 'o']
    alphas = [1, 0.1]
    for i in range(len(scenarios)):

        baselines = np.load(os.path.join(base_path, f'baselines.npy')).flatten()
        errors = np.load(os.path.join(
            base_path, f'{scenarios[i]}_errors.npy')).flatten()

        errors = errors * wavelength * 1e3 / (np.pi * 4)

        if 'het_vel' in scenarios[i]:
            order = np.argsort(baselines)
            baselines = baselines[order]
            gtzero = (baselines > 0)
            baselines = baselines[gtzero]
            errors = errors[order][gtzero]
            errors = np.unwrap(errors)

            ax.plot(baselines, errors, marker=scatter_styles[i], alpha=alphas[i],
                    label=f'{scenarioNames[i]}', color=colors[i])
        else:
            ax.scatter(baselines[baselines > 0], errors[baselines > 0], 30, marker=scatter_styles[i], alpha=alphas[i],
                    label=f'{scenarioNames[i]}', color=colors[i])
            
        ax.set_ylabel('Deformation Error [$\mathrm{mm}$]', fontsize=fontsize)
        ax.set_xlabel('Temporal Baseline [$\mathrm{days}$]', fontsize=fontsize)
        ax.legend(loc='best', fontsize=fontsize, framealpha=0.2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.tight_layout()
    plt.savefig(
        './experiments/figures/output/absolute_error.png', dpi=300, transparent=True)
    plt.show()


def plot_maxb():
    fontsize = 13
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
    bws = np.load(os.path.join(base_path, 'bws.npy'))
    print(bws)

    linestyles = ['-', '--']

    ax = axes
    for i in range(len(scenarios)):
        bias = np.load(os.path.join(
            base_path, f'{scenarios[i]}_bias.npy'))
        print(bias.shape)

        ax.plot(bws/30, bias, linestyles[i],
                label=f'{scenarioNames[i]}', color=colors[i], linewidth=4)

    ax.legend(loc='best', fontsize=fontsize)
    ax.set_xlabel(r'Maximum Temporal Baseline [$\mathrm{yr}$]', fontsize=fontsize)
    ax.set_ylabel(r'Bias [$\mathrm{mm}/\mathrm{yr}$]', fontsize=fontsize)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    plt.tight_layout()

    plt.savefig(
        './experiments/figures/output/bias_rate.png', dpi=300, transparent=True)
    plt.show()


def run(P = 90, interval=1, wavelength=0.056):
    plot_maxb()
    plot_rawerror(wavelength=wavelength)

if __name__ == '__main__':
    run()