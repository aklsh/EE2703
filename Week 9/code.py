'''
------------------------------------
Assignment 9 - EE2703 (Jan-May 2020)
Done by Akilesh Kannan (EE18B122)
Created on 20/03/20
Last Modified on 20/03/20
------------------------------------
'''

# Imports
import cmath
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from scipy.linalg import lstsq

# Global Variables
plotsDir = 'plots/'
PI = np.pi
figNum = 0
showAll = False

# Functions used
def hammingWindow(n):
    '''
                    0.54 + 0.46*cos(2πn/(N−1)), |n| <= (N-1)/2
        w[n] =
                    0, otherwise
    '''

    N = n.size
    window = np.zeros(N)
    window = 0.54 + 0.46*np.cos((2*PI*n)/(N-1))
    return fft.fftshift(window)

def plotSpectrum(figTitle, w, Y, xLimit=None, yLimit=None, showFig=False, saveFig=True):
    global figNum
    plt.figure(figNum)
    plt.suptitle(figTitle)
    plt.subplot(211)
    plt.plot(w, abs(Y), lw=2)
    plt.ylabel(r"$\|Y\|$")
    if (xLimit):
        plt.xlim(xLimit)
    if (yLimit):
        plt.ylim(yLimit)
    plt.subplot(212)
    plt.plot(w, np.angle(Y), "ro", lw=2)
    plt.xlim(xLimit)
    plt.ylabel(r"$\angle Y$")
    plt.xlabel(r"$\omega\ \to$")

    if(saveFig):
        plt.savefig(plotsDir + "Fig"+str(figNum)+".png")
    if(showFig):
        plt.show()
    else:
        plt.show(block=False)
    figNum+=1

# Example 1 - sin(sqrt(2)t)

    ## Without windowing

t = np.linspace(-PI, PI, 65)[:-1]
dt = t[1]-t[0]
fmax = 1/dt
y = np.sin(cmath.sqrt(2)*t)
y[0] = 0
y = fft.fftshift(y)
Y = fft.fftshift(fft.fft(y))/64.0
w = np.linspace(-PI*fmax, PI*fmax, 65)[:-1]
plotSpectrum(r"Spectrum of $sin(\sqrt{2}t)$", w, Y, [-10, 10], showFig=showAll)

    ## Windowing with Hamming Window

t = np.linspace(-PI, PI, 65)[:-1]
dt = t[1]-t[0]
fmax = 1/dt
n = np.arange(64)
y = np.sin(cmath.sqrt(2)*t) * hammingWindow(n)
y[0] = 0
y = fft.fftshift(y)
Y = fft.fftshift(fft.fft(y))/64.0
w = np.linspace(-PI*fmax, PI*fmax, 65)[:-1]
plotSpectrum(r"Spectrum of $sin(\sqrt{2}t) * w(t)$", w, Y, [-8, 8], showFig=showAll)


# Question 2 - spectrum of (cos(0.86 t))**3

    ## Without windowing

t = np.linspace(-4*PI, 4*PI, 257)[:-1]
dt = t[1]-t[0]
fmax = 1/dt
y = np.cos(0.86*t)**3
y[0] = 0
y = fft.fftshift(y)
Y = fft.fftshift(fft.fft(y))/256.0
w = np.linspace(-PI*fmax, PI*fmax, 257)[:-1]
plotSpectrum(r"Spectrum of $cos^3(0.86t)$", w, Y, [-8, 8], showFig=showAll)

    ## Windowing with Hamming Window

t = np.linspace(-4*PI, 4*PI, 257)[:-1]
dt = t[1]-t[0]
fmax = 1/dt
n = np.arange(256)
y = (np.cos(0.86*t))**3 * hammingWindow(n)
y[0] = 0
y = fft.fftshift(y)
Y = fft.fftshift(fft.fft(y))/256.0
w = np.linspace(-PI*fmax, PI*fmax, 257)[:-1]
plotSpectrum(r"Spectrum of $cos^3(0.86t) * w(t)$", w, Y, [-8, 8], showFig=showAll)


# Question 3 - Estimation of w, d in cos(wt + d)

wo = 1.35
d = PI/2
t = np.linspace(-PI, PI, 129)[:-1]
trueCos = np.cos(wo*t + d)
fmax = 1.0/(t[1]-t[0])
n = np.arange(128)
y = trueCos.copy()*hammingWindow(n)
y = fft.fftshift(y)
Y = fft.fftshift(fft.fft(y))/128.0
w = np.linspace(-PI*fmax, PI*fmax, 129)[:-1]
plotSpectrum(r"Spectrum of $cos^3(\omega_o t + \delta) \cdot w(t)$", w, Y, [-4, 4], showFig=showAll)

def estimateWandD(w, wo, Y, do):
    wEstimate = np.sum(abs(Y)**2 * abs(w))/np.sum(abs(Y)**2) # weighted average
    print("wo = {:.03f}\t\two (Estimated) = {:.03f}".format(wo, wEstimate))

    t = np.linspace(-PI, PI, 129)[:-1]
    y = np.cos(wo*t + do)

    c1 = np.cos(wEstimate*t)
    c2 = np.sin(wEstimate*t)
    A = np.c_[c1, c2]
    vals = lstsq(A, y)[0]
    dEstimate = np.arctan2(-vals[1], vals[0])
    print("do = {:.03f}\t\tdo (Estimated) = {:.03f}".format(do, dEstimate))

estimateWandD(w, wo, Y, d)
