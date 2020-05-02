'''
------------------------------------
Assignment 10 - EE2703 (Jan-May 2020)
Done by Akilesh Kannan (EE18B122)
Created on 02/05/20
Last Modified on 02/05/20
------------------------------------
'''

# Imports
import csv
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import scipy.signal as sgnl

# Global Variables
plotsDir = 'plots/'
PI = np.pi
figNum = 0
showAll = True

# Helper Functions
def plotSignal(t, x, figTitle=None, style='b-', blockFig=False, showFig=True, saveFig=True, stemPlot=False, xLimit=None, yLimit=None):
    global figNum
    plt.figure(figNum)
    plt.title(figTitle)
    plt.grid()
    if(stemPlot):
        plt.stem(t, x, linefmt='b-', markerfmt='bo')
    else:
        plt.plot(t, x, style)
    if(xLimit):
        plt.xlim(xLimit)
    if(yLimit):
        plt.ylim(yLimit)
    if(saveFig):
        plt.savefig(plotsDir + "Fig"+str(figNum)+".png")
    if(showFig):
        plt.show(block=blockFig)
    figNum+=1


def plotSpectrum(w, Y, figTitle=None, magStyle='b-', phaseStyle='ro', xLimit=None, yLimit=None, showFig=False, saveFig=True, blockFig=False, type="Y"):
    global figNum
    plt.figure(figNum)
    plt.suptitle(figTitle)
    plt.subplot(211)
    plt.grid()
    plt.plot(w, abs(Y), magStyle, lw=2)
    plt.ylabel(r"$\| "+type+"\|$")
    if (xLimit):
        plt.xlim(xLimit)
    if (yLimit):
        plt.ylim(yLimit)
    plt.subplot(212)
    plt.grid()
    plt.plot(w, np.angle(Y), phaseStyle, lw=2)
    plt.xlim(xLimit)
    plt.ylabel(r"$\angle "+type+"$")
    plt.xlabel(r"$\omega\ \to$")

    if(saveFig):
        plt.savefig(plotsDir + "Fig"+str(figNum)+".png")
    if(showFig):
        plt.show(block=blockFig)
    figNum+=1

# Question 1
filter = np.genfromtxt("h.csv")

# Question 2
plotSignal(range(len(filter)), filter, "FIR Filter ($h[n]$)", stemPlot=True, showFig=showAll)
w, H = sgnl.freqz(filter, 1)
plotSpectrum(w, H, "Frequency Response of FIR Filter ($H(e^{j\omega}))$", type="H", showFig=showAll)

# Question 3
n = np.linspace(1, 2**10, 2**10)
x = np.cos(0.2*PI*n) + np.cos(0.85*PI*n)
plotSignal(n, x, figTitle="$x[n] = cos(0.2\pi n) + cos(0.85\pi n)$", xLimit=[0, 50], stemPlot=True, showFig=showAll)

# Question 4
y = np.convolve(x, filter)
plotSignal(list(range(len(y))), y, figTitle=r"$y[n] = x[n]\ast h[n]$", saveFig=True, xLimit=[0, 100], stemPlot=True, showFig=showAll, blockFig=False)

# Question 5
numZeros = len(x)-len(filter)
y = fft.ifft(fft.fft(x)*fft.fft(np.concatenate((filter, np.zeros(numZeros,)))))
plotSignal(list(range(len(y))), y, figTitle=r"$y[n] = x[n]\otimes h[n]$ (N = 1024)", saveFig=True, xLimit=[0, 100], stemPlot=True, showFig=showAll, blockFig=False)

# Question 6
numZerosForX = len(filter) - 1
numZerosForH = len(x) - 1
paddedX = np.concatenate((x, np.zeros(numZerosForX,)))
paddedH = np.concatenate((filter, np.zeros(numZerosForH,)))
y = fft.ifft(fft.fft(paddedX)*fft.fft(paddedH))
plotSignal(list(range(len(y))), y, figTitle=r"$y[n] = x[n]\otimes h[n]$ (N = 1034), with zero-padding of $x[n]$ and $h[n]$", saveFig=True, xLimit=[0, 100], stemPlot=True, showFig=showAll, blockFig=False)

# Question 7
def readComplexNumbers(fileName):
    rawLines = []
    actualValues = []
    with open(fileName, "r") as p:
        rawLines = p.readlines()
    for line in rawLines:
        actualValues.append(complex(line))
    return actualValues

zChu = readComplexNumbers("x1.csv")
zChuShifted = np.roll(zChu, 5)
y = fft.ifftshift(np.correlate(zChuShifted, zChu, "full"))
plotSignal(list(range(len(y))), abs(y), figTitle=r"Zadoff-Chu Sequence", saveFig=True, stemPlot=True, showFig=showAll, blockFig=True)

