"""Blah."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.special import jv
from astropy.constants import k_B
from math import factorial
k_B = k_B.value



def plot_temps():

    t_ants = [326.1, 291.59, 259.3, 292.0]
    times = [1.5, 4, 0, 6]
    ambient_temps = [295.928, 297.594, 259.3, 296.]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    ax1.plot(times, t_ants, 'or')
    ax2.plot(ambient_temps, t_ants, 'ob')

    ax1.set_xlabel('Times (Hours after 9:30 AM)', weight='bold')
    ax2.set_xlabel('Ambient Temps (K)', weight='bold')
    ax1.set_ylabel('Temperature (K)', weight='bold')

    fig.subplots_adjust(wspace=0)

    plt.savefig('sys_temps.pdf')
    plt.show(block=False)
    plt.gca()


gamma = 1
D = 2.4
lam = 0.21
A_g = np.pi * (D/2)**2

def G_old(thetas, func='airy', gamma=1):
    """docstring."""
    # Gain function:
    first_term = (2**(gamma + 1) * factorial(gamma + 1))**2
    second_term = 4 * np.pi * A_g * lam**(-2)
    x = np.pi * D * thetas/lam
    third_term = (jv(gamma + 1, x)/x**2)**2
    gain = first_term * second_term * third_term
    return gain


def G(thetas, func='airy', gamma=1):
    """docstring."""
    # Gain function:
    first_term = 48 * (np.pi * D/lam)**2
    x = np.pi * D * thetas/lam
    second_term = (jv(gamma + 1, x)/x**2)**2
    gain = first_term * second_term
    return gain


def airy(thetas):
    coeff = (2 * np.pi * D/lam)**2
    xs = np.pi * thetas * D / lam
    airy = coeff * (jv(1, xs)/xs)**2
    return airy


def plot_power(subplots=False, alt0=48):
    """This needs work."""

    c_factor = 1./0.6

    d_az = pd.read_csv('power_az.csv', sep=',')
    d_alt = pd.read_csv('power_alt.csv', sep=',')

    az_vals = d_az['Az'] - d_az['Az'][len(d_az['Az'])/2 - 2]
    alt_vals = d_alt['Alt'] - d_alt['Alt'][len(d_alt['Alt'])/2 - 1]

    p_az_normed =            d_az['Power'] #- d_az['Power'][0]
    p_alt_normed =           d_alt['Power'] #- d_alt['Power'][0]

    T_alt = p_alt_normed * c_factor
    T_az = p_az_normed * c_factor


    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6), sharey=True)

    ax1.plot(d_az['Az'], T_az)
    ax2.plot(d_az['Az'] * np.cos(alt0), T_az)
    ax2.plot(d_az['Az'] * np.cos(alt0), [(max(T_az) + min(T_az))/2]*len(d_az['Az']), ':k', label='FWHM')
    ax3.plot(d_alt['Alt'], T_alt)
    ax3.plot(d_alt['Alt'], [(max(T_alt) + min(T_alt))/2]*len(d_alt['Alt']), ':k', label='FWHM')

    ax2.axvline(x=-217.5, color='black', linestyle=':')
    ax2.axvline(x=-212.5, color='black', linestyle=':')
    ax3.axvline(x=45, color='black', linestyle=':')
    ax3.axvline(x=50.8, color='black', linestyle=':')

    ax1.set_xlabel('Az')
    ax2.set_xlabel('Az')
    ax2.set_xlabel('Alt')
    ax1.set_ylabel('Antenna Temperature')
    ax1.set_title('Azimuthal Ant. Temp (Raw)', weight='bold')
    ax2.set_title('Azimuthal Ant. Temp (Alt-Corrected)', weight='bold')
    ax3.set_title('Altitudinal Ant. Temp', weight='bold')
    fig.subplots_adjust(wspace=0)

    plt.legend()
    plt.savefig('part1_1.png', dpi=200)
    plt.show(block=False)
    plt.gca()


    thetas = np.arange(az_vals[0], az_vals[len(az_vals)-1], 0.001) * np.pi/180
    fig2, (ax4, ax5) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    ax4.plot(az_vals,
             T_az - T_az[0],
             label='Raw Power (Az)')
    ax4.plot(az_vals * np.cos(alt0),
             T_az - T_az[0],
             label='Alt-Corrected Power (Az)')
    ax4.plot(alt_vals,
             T_alt - T_alt[0],
             label='Power (Alt)')
    ax4.legend()

    gs = G_old(thetas)
    airs = airy(thetas)
    ax5.plot(thetas,
             gs, label='Non-Uniform\nIllumination')
    ax5.plot(thetas,
             airs, label='Uniform Illumination\n(Airy Fn)')
    ax5.plot(thetas, [max(gs)/2]*len(thetas), ':k', label='FWHM')

    ax5.axvline(x=-0.06, color='black', linestyle=':')
    ax5.axvline(x=0.06, color='black', linestyle=':')

    ax5.legend()
    ax4.set_xlabel('Angular offset (Degrees)')
    ax5.set_xlabel('Angular offset (Radians)')
    ax4.set_title('Observed Ant. Temps, Compared', weight='bold')
    ax5.set_title('Theoretical Ant. Temps, Compared', weight='bold')

    #ax5.plot(az_vals * np.cos(alt0),
    #         T_az_corrected - T_az_corrected[0],
    #         label='Alt-Corrected Power (Az)')
    fig2.subplots_adjust(wspace=0)

    plt.savefig('part1_2.png', dpi=200)
    plt.show(block=False)
    plt.gca()


#plot_power()


def power_stuff1():
    """Power stuff."""
    F_casa = 2000               # Janskies
    df = pd.read_csv('cas_a.csv', sep=',')
    p_on = list(df['on_r'].dropna())
    p_off = list(df['off_r_159'].dropna())

    P_ave = np.mean(p_on[5:]) - np.mean(p_off[5:])
    return P_ave


def power_stuff():
    """Power stuff."""
    F_casa = 2000               # Janskies
    df = pd.read_csv('cas_a.csv', sep=',')
    p_on = list(df['on_r'].dropna())
    p_off = list(df['off_r_159'].dropna())

    N = len(p_on)

    p = [p_on[i] - p_off[i] for i in range(len(p_on))][3:]
    p_ave = np.mean(p)

    return P_ave






def part_3(r_or_l='r'):
    """This kinda works I guess?"""
    if r_or_l == 'r':
        N = N_r
        P_ons = on_r
        P_offs = off_r_159
    elif r_or_l == 'l':
        N = N_l
        P_ons = on_l
        P_offs = off_l_119
    else:
        return "Choose r or l"

    P_ks = []
    for k in range(1, N):
        # Odds:
        if k % 2 == 1:
            P_k = P_ons[k] - P_offs[k-1]
        # Evens
        else:
            P_k = P_ons[k-1] - P_offs[k]

        P_ks.append(P_k)

    # print P_ks
    P_ave = np.sum(P_ks)/N
    # print P_ave
    P_sd = np.sqrt(np.sum(P_ks)**2 - N * P_ave**2) / np.sqrt(N - 1)
    return P_ave


# The End
