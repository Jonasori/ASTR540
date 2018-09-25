import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from astropy.constants import h, c, k_B, R_earth
h, c, k = h.value, c.value, k_B.value

nu_min = 1e7
nu_max = 1e12

def Tb(nu):
    return 180 * (nu/(180e6))**(-2.6) + 2.7

def Planck(nu):
    I = (2 * h * nu**3 * c**-2) \
        * 1/(np.exp((h * nu)/(k * Tb(nu))) - 1)
    return I

def Energy(n_years=1):
    hours = 8760 * n_years
    I = integrate.quad(Planck, nu_min, nu_max)[0]
    P = (4 * np.pi * R_earth.value)**2 * I
    e = P * hours
    # Make sure to return in TWh
    return e * 1e-12

def plot_planck():
    nus = np.arange(1e7, 1e12, 1e8)
    plancks = Planck(nus)
    plt.plot(nus, plancks, '-b')
    plt.plot()
    plt.fill_between(nus, 0, plancks, color='blue', alpha=0.3)
    plt.title('Planck Function of CMB and Synchrotron Radiation \n in the 1MHz-1THz Band')
    plt.ylabel('Specific Intensity')
    plt.xlabel('Frequency (Hz)')
    plt.savefig('Planck_func.png')
