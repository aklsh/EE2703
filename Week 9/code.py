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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Global Variables
plotsDir = 'plots/'
PI = np.pi
figNum = 0
showAll = True

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

def plotSignal(t, x, figTitle, style='b-', blockFig=False, showFig=True, saveFig=False):
    global figNum
    plt.figure(figNum)
    plt.title(figTitle)
    plt.grid()
    plt.plot(t, x, style)
    if showFig:
        plt.show(block=blockFig)
    if saveFig:
        plt.savefig(plotsDir + "Fig"+str(figNum)+".png")
    figNum+=1


def plotSpectrum(figTitle, w, Y, magStyle='b-', phaseStyle='ro', xLimit=None, yLimit=None, showFig=False, saveFig=True, blockFig=False):
    global figNum
    plt.figure(figNum)
    plt.suptitle(figTitle)
    plt.subplot(211)
    plt.grid()
    plt.plot(w, abs(Y), magStyle, lw=2)
    plt.ylabel(r"$\|Y\|$")
    if (xLimit):
        plt.xlim(xLimit)
    if (yLimit):
        plt.ylim(yLimit)
    plt.subplot(212)
    plt.grid()
    plt.plot(w, np.angle(Y), phaseStyle, lw=2)
    plt.xlim(xLimit)
    plt.ylabel(r"$\angle Y$")
    plt.xlabel(r"$\omega\ \to$")

    if(saveFig):
        plt.savefig(plotsDir + "Fig"+str(figNum)+".png")
    if(showFig):
        plt.show(block=blockFig)
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
plotSpectrum(r"Spectrum of $sin(\sqrt{2}t)$", w, Y, xLimit=[-10, 10], showFig=showAll)

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
plotSpectrum(r"Spectrum of $sin(\sqrt{2}t) * w(t)$", w, Y, xLimit=[-8, 8], showFig=showAll)


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
plotSpectrum(r"Spectrum of $cos^3(0.86t)$", w, Y, xLimit=[-8, 8], showFig=showAll)

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
plotSpectrum(r"Spectrum of $cos^3(0.86t) * w(t)$", w, Y, xLimit=[-8, 8], showFig=showAll)


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


# Question 3 - Estimation of w, d in cos(wt + d)
print("Question 3:")
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
plotSpectrum(r"Spectrum of $cos(\omega_o t + \delta) \cdot w(t)$", w, Y, xLimit=[-4, 4], showFig=showAll, saveFig=False)
estimateWandD(w, wo, Y, d)

print("\nQuestion 4:")

# Question 4 - Estimation of w, d in noisy cos(wt + d)

trueCos = np.cos(wo*t + d)
noise = 0.1*np.random.randn(128)
n = np.arange(128)
y = (trueCos + noise)*hammingWindow(n)
fmax = 1.0/(t[1]-t[0])
y = fft.fftshift(y)
Y = fft.fftshift(fft.fft(y))/128.0
w = np.linspace(-PI*fmax, PI*fmax, 129)[:-1]
plotSpectrum(r"Spectrum of $(cos(\omega_o t + \delta) + noise) \cdot w(t)$", w, Y, xLimit=[-4, 4], showFig=showAll, saveFig=False)
estimateWandD(w, wo, Y, d)

# Question 5 - DFT of chirp

def chirp(t):
    return np.cos(16*(1.5*t + (t**2)/(2*PI)))

t = np.linspace(-PI, PI, 1025)[:-1]
x = chirp(t)
plotSignal(t, x, r"$cos(16(1.5 + \frac{t}{2\pi})t)$")
fmax = 1.0/(t[1]-t[0])
X = fft.fftshift(fft.fft(x))/1024.0
w = np.linspace(-PI*fmax, PI*fmax, 1025)[:-1]
plotSpectrum(r"DFT of $cos(16(1.5 + \frac{t}{2\pi})t)$", w, X, 'b-', 'r.-', [-75, 75], showFig=showAll, saveFig=True)

n = np.arange(1024)
x = chirp(t)*hammingWindow(n)
plotSignal(t, x, r" $cos(16(1.5 + \frac{t}{2\pi})t) \cdot w(t)$")
X = fft.fftshift(fft.fft(x))/1024.0
plotSpectrum(r"DFT of $cos(16(1.5 + \frac{t}{2\pi})t) \cdot w(t)$", w, X, 'b-', 'r.-', [-75, 75], showFig=showAll, saveFig=True)

# Question 6 - Time evolution of DFT of chirp signal

def STFT(chirp, t, batchSize=64):
    '''
        returns 2d array, ready for plotting
    '''
    t_batch = np.split(t, 1024//batchSize)
    x_batch = np.split(x, 1024//batchSize)
    X = np.zeros((1024//batchSize, batchSize), dtype=complex)
    for i in range(1024//batchSize):
        X[i] = fft.fftshift(fft.fft(x_batch[i]))/batchSize
    return X

def plot3DSTFT(t, w, X, colorMap=cm.viridis, showFig=showAll, saveFig=True, blockFig=False):
    global figNum

    t = t[::64]
    w = np.linspace(-fmax*PI,fmax*PI,65)[:-1]
    t, w = np.meshgrid(t, w)

    fig = plt.figure(figNum)
    ax = fig.add_subplot(211, projection='3d')
    surf = ax.plot_surface(w, t, abs(X).T, cmap=colorMap)
    fig.colorbar(surf)
    plt.ylabel(r"Frequency $\to$")
    plt.xlabel(r"Time $\to$")
    plt.title(r"Magnitude $\|Y\|$")

    ax = fig.add_subplot(212, projection='3d')
    surf = ax.plot_surface(w, t, np.angle(X).T, cmap=colorMap)
    fig.colorbar(surf)
    plt.ylabel(r"Frequency $\to$")
    plt.xlabel(r"Time $\to$")
    plt.title(r"Angle $\angle Y$")
    if saveFig:
        plt.savefig(plotsDir+"Fig"+str(figNum)+".png")
    if showFig:
        plt.show(block=blockFig)

    figNum+=1

x = chirp(t)
X = STFT(x, t)
plot3DSTFT(t, w, X, colorMap=cm.plasma)

x = chirp(t)*hammingWindow(np.arange(1024))
X = STFT(x, t)
plot3DSTFT(t, w, X, blockFig=True)
