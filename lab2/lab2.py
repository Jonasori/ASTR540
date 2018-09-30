"""Scratch work for Lab ."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

d1 = pd.read_csv('part1.csv', sep=',')
d2 = pd.read_csv('part2.csv', sep=',')
d3 = pd.read_csv('part3.csv', sep=',')
d4 = pd.read_csv('part4.csv', sep=',')


# Part 1:
thetas_obs = np.arcsin(d1['Baseline']/48.)
# Little d is the width of the CFL (inches)
d = 3


plt.errorbar(d1['Baseline'], d1['temp'], d1['uncertainty'])
plt.errorbar(thetas_obs, d1['temp'], d1['uncertainty'])

def G(alpha, beta=0):
    D = 48
    lam = 1
    gain = (np.sinc(np.pi * d * alpha / lam))**2 * \
           (np.sinc(np.pi * d * beta / lam))**2 * \
           (np.cos(np.pi * D * alpha / lam))**2 * \
           (np.cos(np.pi * D * beta / lam))**2
    return gain

plt.plot(thetas_obs, G(thetas_obs))


# Part 2:
thetas_obs2 = np.arcsin(d2['Baseline']/48.)
plt.errorbar(thetas_obs2, d2['temp'], d2['uncertainty'])





# Part 3:
thetas_obs3 = np.arcsin(d3['Baseline']/48.)
plt.errorbar(thetas_obs3, d3['temp'], d3['uncertainty'])
