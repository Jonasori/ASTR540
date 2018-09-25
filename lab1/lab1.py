"""Blah."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.special import jn, airy
from astropy.constants import k_B
k_B = k_B.value

def plot_power(subplots=False, alt0=48):
    """This needs work."""

    d_az = pd.read_csv('power_az.csv', sep=',')
    d_az_corrected = -1 * d_az['Power'] * np.cos(alt0) #+ d_az['Power'][0]
    d_alt = pd.read_csv('power_alt.csv', sep=',')
    airy_az  = airy(d_az['Az'])[0]
    airy_alt = airy(d_alt['Alt'])[0]

    airy_az
    d_alt['Alt']

    az_vals = d_az['Az'] - d_az['Az'][len(d_az['Az'])/2]
    alt_vals = d_alt['Alt'] - d_alt['Alt'][len(d_alt['Alt'])/2 + 1]
    p_az_normed = d_az['Power'] - d_az['Power'][0]
    p_az__corrected_normed = d_az_corrected - d_az_corrected[0]
    p_alt_normed = d_alt['Power'] - d_alt['Power'][0]



    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 6))
    ax1.plot(d_az['Az'], d_az['Power'])
    ax2.plot(d_az['Az'], d_az_corrected)
    ax3.plot(d_alt['Alt'], d_alt['Power'])

    ax4.plot(az_vals, p_az_normed, label='Raw Power (Az)')
    ax4.plot(az_vals, p_az__corrected_normed, label='Corrected Power (Az)')
    ax4.plot(alt_vals, p_alt_normed, label='Power (Alt)')
    ax4.legend()

    ax1.set_xlabel('Az')
    ax2.set_xlabel('Az')
    ax3.set_xlabel('Alt')
    ax1.set_ylabel('Power')
    ax2.set_ylabel('Power (Corrected)')
    ax3.set_ylabel('Power')
    plt.savefig('part1.png', dpi=200)

    plt.show(block=False)
    plt.gca()


def part_I():
    return blah




def power_stuff(N=100):
    """Power stuff."""
    P_offs = np.zeros(N)
    P_ons = np.zeros(N)

    P_sds = np.zeros(N)
    P_aves = np.zeros(N)
    for k in range(0, N):
        # Odds:
        if k % 2 == 1:
            P_k = P_ons[k] - P_offs[k-1]
        # Evens
        else:
            P_k = -P_offs[k] + P_ons[k-1]

        P_aves[k] = np.sum(P_k)/N
        P_sds[k] = np.sqrt(np.sum(P_k**2) - N * P_aves**2) / np.sqrt(N - 1)


def T(B, P):
    """blah."""
    temp = P/(k_B * B)
    return temp


def eta(T_A):
    """Efficiency stuff."""
    F = 1e-26                   # W Hz-1 K-1
    A_G = 2.4**2
    n = 2 * k_B * T_A / (F * A_G)
    return n


def epsilon(rho):
    """docstring."""
    D = 2.4
    gamma = 1
    e = (1 - (2 * rho/D)**2)**gamma
    return e


def G(theta):
    """docstring."""
    gamma = 1
    D = 2.4
    lam = 0.021
    A_g = 1

    first_term = (2**(gamma + 1) * np.factorial(gamma + 1))**2
    second_term = 4 * np.pi * A_g * lam**(-2)
    x = np.pi * D * theta/lam
    third_term = jn(gamma + 1, x) * x**(-2)
    gain = first_term * second_term * third_term
    return gain



# The End
