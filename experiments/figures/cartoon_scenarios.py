import numpy as np
from matplotlib import pyplot as plt
import figstyle


def dielectric_cartoons(P = 90):
    linecolor = 'darkslategrey'
    imag_color = 'black'
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    x = np.arange(0, P, 1)
    b = 10
    epsilon =  1/10 * x + b
    epsilon_i =  1/100 * x + b

    axs[0].plot(x, epsilon, color=linecolor, linewidth=3, label=r'$\epsilon_r$')
    axs[0].plot(x, epsilon_i, '--',  color=imag_color, linewidth=3, label=r'$\epsilon_i$')
    axs[0].set_ylim([b - 1, epsilon.max() + 2])
    axs[0].legend(loc='upper left', fontsize=20)
    axs[0].set_title('Secular Trend')



    epsilon = np.cos((x / 30) * 2 * np.pi)
    epsilon_i = epsilon/10
    epsilon += b
    epsilon_i +=b


    axs[1].plot(x, epsilon, linewidth=3, color=linecolor, label=r'$\epsilon_r$')
    axs[1].plot(x, epsilon_i, '--',   color=imag_color, linewidth=3, label=r'$\epsilon_i$')
    axs[1].set_ylim([b - 1.5 , epsilon.max() + 0.4])
    axs[1].set_title('Seasonal')

    for ax in axs:
        ax.spines['top'].set_color(None)
        ax.spines['right'].set_color(None)
        ax.set_yticks([])
        ax.set_xlabel('t (aquisition)')
        ax.set_ylabel(r'[$\epsilon$]', fontsize=20)

    plt.tight_layout()
    plt.savefig('./experiments/figures/output/dielectric_cartoons.png', dpi=300,)
    plt.show()


def heterogenous_vel_cartoon(trough_vel = 200, center_vel = 50, N = 10000, trough_p = 0.2, center_p = 0.8):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))

    trough = np.ones(int(N * trough_p)) * trough_vel
    center = np.ones(int(N * center_p)) * center_vel

    velocities = np.concatenate([trough, center])

    print(f'Mean Velocity: {np.mean(velocities)}')
    print(f'Median Velocity: {np.median(velocities)}')

    labels, counts = np.unique(velocities, return_counts=True)
    # plt.axhline(y=trough_p*N, color='gray', linestyle='-', alpha=0.2)
    # plt.axhline(y=center_p*N, color='gray', linestyle='-', alpha=0.2)
    # plt.axhline(y=trough_p*N * 2, color='gray', linestyle='-', alpha=0.2)
    # plt.axhline(y=trough_p*N * 3, color='gray', linestyle='-', alpha=0.2)
    vel_range = np.abs(trough_vel - center_vel)

    plt.bar(labels, counts, width = vel_range/4, align='center', color='darkslategrey', alpha=0.9)

    plt.gca().set_xticks(labels)


    # ax.hist(velocities, bins=5, color='darkslategrey')
    # # ax.legend(loc='upper left', fontsize=20)
    # ax.set_title('Heterogenous Velocities')
    # ax.yaxis.set_major_formatter(lambda x, pos: str(x/N))
    ax.set_yticks([trough_p*N, trough_p*N*2, trough_p*N*3, center_p*N], labels=[trough_p, 0.4, 0.6, center_p])   
    ax.set_xticks([center_vel, trough_vel], labels=[center_vel, trough_vel])    
 
    ax.grid(axis='y', alpha=0.2)

    ax.spines['top'].set_color(None)
    ax.spines['right'].set_color(None)


    ax.set_xlim([center_vel + vel_range/2, trough_vel - vel_range/2])

    ax.set_xlabel(r'Velocity [$\frac{mm}{yr}$]')
    ax.set_ylabel(r'Relative Frequency', fontsize=20)

    plt.tight_layout()
    plt.savefig('./experiments/figures/output/hetVelCartoon.png', dpi=300,)
    plt.show()



def run(P = 90, interval=1, wavelength=0.056, trough_vel=200, center_vel=50, trough_p=0.2, center_p=0.8):
    dielectric_cartoons(P = P)
    heterogenous_vel_cartoon(trough_vel=trough_vel, center_vel=center_vel, trough_p=trough_p, center_p=center_p)

if __name__ == '__main__':
    run(90, wavelength=0.056, interval=1, trough_vel=200, center_vel=50, trough_p=0.2, center_p=0.8)