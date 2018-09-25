"""Some helper functions for HW2."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv
import peakutils
import matplotlib.colors as colors
from matplotlib.colors import SymLogNorm

def problem1():
    D = 10
    l = 0.01

    # Evaluate the power function on a list of phases.
    thetas = np.arange(-0.005, 0.005, 0.00001)
    ps = (np.sinc(thetas * D/l))**2

    # Use PeakUtils to find some peaks.
    # Slice to get the five central peaks.
    idx = peakutils.indexes(ps, thres=0.0001, min_dist=0.001)
    idx = idx[len(idx)/2 - 2:len(idx)/2 + 3]

    # Print out the results, then plot them.
    for i in idx:
        print 'Theta: ', thetas[i]
        print 'Power: ', ps[i]/max(ps)
        print

    plt.plot(thetas, ps, '-b')
    plt.scatter(thetas[idx], ps[idx])
    plt.savefig('problem1.png', dpi=200)


def problem2():
    D = 1
    dD = 0.1
    l = 1
    def Gain(theta, l):
        D = 1
        dD = 0.1
        G = 2 * (np.pi/l)**2 * jv(0, (np.pi * D/l) * theta) * D * dD
        return G


    thetas = np.arange(-np.pi/2, np.pi/2, 0.01)
    gs = Gain(thetas, l)
    plt.plot(thetas, gs)
    plt.yticks([])
    plt.xlabel('Theta (radians)')
    plt.ylabel('Gain')
    plt.savefig('problem2a.png', dpi=200)
    plt.gca()



def problem3(vmax=5000):
    # Begin by setting up the aperture
    crossbar_width = 3
    r_circle = 10
    r_big_circle = 50
    p0 = 10

    def a(x, y):
        if np.sqrt(x**2 + y**2) <= r_circle:
            p = 0
        elif abs(x) <= crossbar_width or abs(y) <= crossbar_width:
            p = 0
        elif np.sqrt(x**2 + y**2) > r_big_circle:
            p = 0
        else:
            p = p0
        return p


    xs = np.arange(-1. * r_big_circle,
                   1. * r_big_circle,
                   0.05)
    ys = np.arange(-1. * r_big_circle,
                   1. * r_big_circle,
                   0.05)

    apertures = np.zeros((len(xs), len(ys)))

    for x in range(len(ys)):
        for y in range(len(ys)):
            apertures[x, y] = a(xs[x], ys[y])

    ft_real = np.real(np.fft.fft2(apertures))
    ft_abs = np.abs(np.fft.fft2(apertures))


    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(apertures)
    ax1.set_xticks([])
    ax1.set_yticks([])
    cax2 = ax2.imshow(ft_abs,
                      vmin=np.min(ft_abs),
                      vmax=vmax)

    fig.colorbar(cax2, fraction=0.046, pad=0.04)
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.savefig('problem3.png')
    #plt.show(block=False)
    plt.gca()

    # For log color:
    #ax2.pcolor(ft, norm=SymLogNorm(linthresh=0.1, linscale=0.1, vmin=np.min(ft), vmax=np.max(ft)), cmap='PuBu_r')






# The End
