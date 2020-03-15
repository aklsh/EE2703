'''
------------------------------------
Assignment 8 - EE2703 (Jan-May 2020)
Done by Akilesh Kannan (EE18B122)
Created on 15/03/20
Last Modified on 15/03/20
------------------------------------
'''

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt

plotsDir = 'plots/'
PI = np.pi

## Random Sequence FFT and IFFT
xOriginal = np.random.rand(128)
X = fft.fft(xOriginal)
xComputed = fft.ifft(X)
plt.figure(0)
t = np.linspace(-64, 63, 128)
plt.plot(t, xOriginal, 'b', label='Original $x(t)$')
plt.plot(t, np.abs(xComputed), 'g', label='Computed $x(t)$')
plt.xlabel(r'$t\ \to$')
plt.grid()
plt.legend()
plt.title('Comparison of actual and computed $x(t)$')
plt.savefig(plotsDir+'Fig0.png')
maxError = max(np.abs(xComputed-xOriginal))
print(r'Magnitude of maximum error between actual and computed values: ', maxError)     # order of 1e-15


## Spectrum of sin(5t)
x = np.linspace(0, 2*PI, 129)
x = x[:-1]
y = np.sin(5*x)
Y = fft.fftshift(fft.fft(y))/128.0
fig1 = plt.figure(1)
fig1.suptitle(r'FFT of $sin(5t)$')
YMag = np.abs(Y)
YPhase = np.angle(Y)
w = np.linspace(-64, 63, 128)
plt.subplot(211)
plt.plot(w, YMag)
plt.xlim([-10, 10])
plt.ylabel(r'$\|Y\|$')
plt.grid()
plt.subplot(212)
plt.plot(w, YPhase, 'ro')
plt.xlim([-10, 10])
plt.ylabel(r'$\angle Y$')
plt.xlabel(r'$k\ \to$')
presentFreqs = np.where(YMag > 1e-3)
plt.plot(w[presentFreqs], YPhase[presentFreqs], 'go')
plt.grid()
plt.savefig(plotsDir+'Fig1.png')


## AM Modulation with (1 + 0.1cos(t))cos(10t)
x = np.linspace(-4*PI, 4*PI, 513)
x = x[:-1]
y = (1+0.1*np.cos(x))*np.cos(10*x)
Y = fft.fftshift(fft.fft(y))/512.0
fig2 = plt.figure(2)
fig2.suptitle(r'AM Modulation with $(1+0.1cos(t))cos(10t)$')
YMag = np.abs(Y)
YPhase = np.angle(Y)
w = np.linspace(-64, 64, 512)
plt.subplot(211)
plt.plot(w, YMag)
plt.xlim([-15, 15])
plt.ylabel(r'$\|Y\|$')
plt.grid()
plt.subplot(212)
plt.plot(w, YPhase, 'ro')
plt.xlim([-15, 15])
plt.ylabel(r'$\angle Y$')
plt.xlabel(r'$k\ \to$')
presentFreqs = np.where(YMag > 1e-3)
plt.plot(w[presentFreqs], YPhase[presentFreqs], 'go')
plt.grid()
plt.savefig(plotsDir+'Fig2.png')


## Spectrum of sin^3(t)
x = np.linspace(-4*PI, 4*PI, 513)
x = x[:-1]
y = (np.sin(x))**3
Y = fft.fftshift(fft.fft(y))/512.0
fig3 = plt.figure(3)
fig3.suptitle(r'Spectrum of $sin^3(t)$')
YMag = np.abs(Y)
YPhase = np.angle(Y)
w = np.linspace(-64, 64, 513)
w = w[:-1]
plt.subplot(211)
plt.plot(w, YMag)
plt.xlim([-5, 5])
plt.ylabel(r'$\|Y\|$')
plt.grid()
plt.subplot(212)
plt.plot(w, YPhase, 'ro')
plt.xlim([-5, 5])
plt.ylabel(r'$\angle Y$')
plt.xlabel(r'$k\ \to$')
plt.savefig(plotsDir+'Fig3.png')


## Spectrum of cos^3(t)
x = np.linspace(-4*PI, 4*PI, 513)
x = x[:-1]
y = (np.cos(x))**3
Y = fft.fftshift(fft.fft(y))/512.0
fig4 = plt.figure(4)
fig4.suptitle(r'Spectrum of $cos^3(t)$')
YMag = np.abs(Y)
YPhase = np.angle(Y)
w = np.linspace(-64, 64, 513)
w = w[:-1]
plt.subplot(211)
plt.plot(w, YMag)
plt.xlim([-5, 5])
plt.ylabel(r'$\|Y\|$')
plt.grid()
plt.subplot(212)
plt.plot(w, YPhase, 'ro')
plt.xlim([-5, 5])
plt.ylabel(r'$\angle Y$')
plt.xlabel(r'$k\ \to$')
plt.savefig(plotsDir+'Fig4.png')


## Spectrum of cos(20t + 5cos(t))
x = np.linspace(-4*PI, 4*PI, 513)
x = x[:-1]
y = np.cos(20*x + 5*np.cos(x))
Y = fft.fftshift(fft.fft(y))/512.0
fig5 = plt.figure(5)
fig5.suptitle(r'Spectrum of $cos(20t + 5cos(t))$')
YMag = np.abs(Y)
YPhase = np.angle(Y)
w = np.linspace(-64, 64, 513)
w = w[:-1]
plt.subplot(211)
plt.plot(w, YMag)
plt.xlim([-50, 50])
plt.ylabel(r'$\|Y\|$')
plt.grid()
plt.subplot(212)
significantPhase = np.where(YMag > 1e-3)
plt.plot(w[significantPhase], YPhase[significantPhase], 'ro')
plt.xlim([-50, 50])
plt.ylabel(r'$\angle Y$')
plt.xlabel(r'$k\ \to$')
plt.savefig(plotsDir+'Fig5.png')


## Spectrum of Gaussian





plt.show()
