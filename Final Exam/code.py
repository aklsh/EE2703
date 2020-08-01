"""
-----------------------------------------------
EE2703: Applied Programming Lab (Jan-May 2020)

Assignment: Final Exam
Name: Akilesh Kannan
Roll no.: EE18B122
-----------------------------------------------
"""

# imports
import sys
from scipy.linalg import lstsq
import numpy as np
import matplotlib.pyplot as plt

# Define constants in SI units, wherever applicable
Lx = 0.1        # width of tank
Ly = 0.2        # height of tank
Eo = 8.85e-12   # permittivity of free space
Er = 2          # dielectric constant of water
L = 1           # inductance of external inductor in H(enry)
wo = 2*np.pi    # resonant frequency of circuit

saveAll = True
showAll = True

def findExpFit(errors, iterations, printFit=False):
    '''
    Find LSTSQ Fit (exponential) for
    x = iteration, y = error

                   Bx
            y = A.e
              (or)
    log(y) = log(A) + Bx


    Input
    -----
    errors: list/numpy 1d array
        error vector
    iterations: list/numpy 1d array
        iteration vector

    Output
    ------
    fit: numpy 1d array
        coefficients A, B
    estimate: numpy 1d array
        estimated y values

    '''
    # get number of x-values
    nRows = len(errors)

    # initialise coeffMatrix and constMatrix
    coeffMatrix = np.zeros((nRows,2), dtype=float)
    constMatrix = np.zeros_like(errors)

    # coeffMatrix = [1, iterations]
    coeffMatrix[:,0] = 1
    coeffMatrix[:,1] = iterations

    # constMatrix = log(errors)
    constMatrix = np.log(errors)

    # fit
    fit = lstsq(coeffMatrix, constMatrix)[0]

    # debug statements
    if printFit==True:
        print("LSTSQ Fit parameters")
        print("--------------------")
        print("logA =", fit[0])
        print("B =", fit[1])

    estimate = coeffMatrix@fit
    return fit, estimate

def partD(M, N, step, k, accuracy, No):
    '''
    Function to solve Laplace's Equation
    in the tank.

    Assumes that top of tank is at 1V.

    Input
    -----
    M: int
        number of nodes along X-axis, including
        boundary nodes
    N: int
        number of nodes along Y-axis, including
        boundary nodes
    step: float
        distance between nodes (assumed same for
        X- and Y- axes)
    k: int
        index corresponding to height h
    accuracy: float
        desired accuracy
    No: int
        maximum number of iterations

    Output
    ------
    phi: 2d numpy array (MxN)
        array of solved potentials
    N: int
        number of iterations carried out
    err: 1d numpy array
        error vector

    '''

    # initialise potentials to 0 everywhere, except at top plate
    # potential at top = 1V
    phi = np.zeros((N, M), dtype=float)
    phi[-1, :] = 1.0

    # create meshgrid for plotting potential distribution and for later
    # calculation of Electric field
    x = np.linspace(0, Lx, M, dtype=float)
    y = np.linspace(0, Ly, N, dtype=float)
    X, Y = np.meshgrid(x, y)

    plotContour(X, Y, phi, figTitle='Initial potential distribution')
    iteration=[]     # iteration number
    error=[]         # error vector

    # iteratively calculate potentials
    for i in range(No):
        # create copy of potentials
        oldPhi = phi.copy()

        # updating the potentials
        phi[1:-1, 1:-1] = 0.25*(phi[1:-1, 0:-2]+phi[1:-1, 2:]+phi[0:-2, 1:-1]+phi[2:, 1:-1])
        phi[k, 1:-1] = (Er*phi[k+1, 1:-1] + phi[k-1, 1:-1])*1.0/(1+Er)

        # Applying Boundary Conditions
        phi[0, :] = 0.0          # bottom edge
        phi[:, -1] = 0.0         # right edge
        phi[:, 0] = 0.0          # left edge
        phi[-1, :] = 1.0         # top edge

        # calculating error
        currError = np.abs(phi-oldPhi).max()
        error.append(currError)
        iteration.append(i)

        # stop if accuracy reached
        if currError <= accuracy:
            break

    plotContour(X, Y, phi, figTitle='Potential distribution after updating')

    # find LSTSQ Estimate for exponential region (>5000 iterations)
    fit, estimate = findExpFit(error[5000:], iteration[5000:], printFit=True)

    # extrapolate the estimated error function till iteration 0
    estimate = np.e**(fit[0]+np.multiply(fit[1], iteration))

    plotSemilogy([iteration, iteration], [error, estimate], multiplePlots=True, labels=["Actual error", "Fitted error (iteration >= 5000)"], figTitle='Error vs. iteration', xLabel=r"iterations $\to$", yLabel=r'error $\to$')

    # calculate E
    Ex, Ey = findEField(phi, step, M, N)

    # calculate charge densities
    sigma = findSigma(Ex, Ey, k)

    # calculate charges Qtop and Qfluid
    Q = findCharges(sigma, k, step)

    return phi, Q, iteration[-1], error, Ex, Ey, sigma

def findEField(phi, step, M, N):
    '''
    Calculates the x- and y- components of E-field at
    each point.

    Input
    -----
    phi: 2d numpy array
        potential array
    step: float
        distance between 2 points on the grid
    X: 2d numpy array
        meshgrid X-coordinates
    Y: 2d numpy array
        meshgrid Y-coordinates

    Output
    ------
    Ex: 2d numpy array
        X-components of E field
    Ey: 2d numpy array
        Y-components of E-field

    '''
    # Ex calculation
    #   *   *   *     row i
    #     -   -
    #   *   *   *     row i+1
    #
    negativeGradientX = (phi[:, :-1] - phi[:, 1:])*(1.0/step)
    Ex = (negativeGradientX[:-1, :] + negativeGradientX[1:, :])*0.5

    # Ey calculation
    #   *       *
    #       -
    #   *       *
    #       -
    #   *       *
    # col i   col i+1
    #
    negativeGradientY = (phi[:-1, :] - phi[1:, :])*(1.0/step)
    Ey = (negativeGradientY[:, :-1] + negativeGradientY[:, 1:])*0.5

    # plot
    x = np.linspace(0, Lx, M-1, dtype=float)
    y = np.linspace(0, Ly, N-1, dtype=float)
    X, Y = np.meshgrid(x, y)
    plotQuiver(X, Y, Ex, Ey, r"Vector Plot of $\vec{E}$", blockFig=False)

    return Ex, Ey

def findSigma(Ex, Ey, k):
    '''
    Find the charge density (linear) on
    each side of the tank

    Input
    -----
    Ex: 2d numpy array
        X-component of Electric field at all
        points inside the tank
    Ey: 2d numpy array
        Y-component of Electric field at all
        points inside the tank
    k: int
        index corresponding to boundary

    Output
    ------
    sigma: list
        [top, right, bottom, left] plate charge
        densities

    '''
    # finding sigma on top plate
    # NOTE: -ve sign due to outward normal
    #       for conductor, which is along
    #       -y direction
    sigmaTop = -Ey[-1, :]*Eo

    # finding sigma on bottom plate
    sigmaBottom = Ey[0, :]*Eo*Er

    # finding sigma on left plate
    # NOTE: for nodes below boundary,
    #       permittivity is Eo*Er
    sigmaLeft = Ex[:, 0]*Eo
    sigmaLeft[:k] = Ex[:k, 0]*Eo*Er

    # finding sigma on right plate
    # NOTE: -ve sign due to outward
    #       normal in -x direction
    # NOTE: for nodes below boundary,
    #       permittivity is Eo*Er
    sigmaRight = -Ex[:, -1]*Eo
    sigmaRight[:k] = -Ex[:k, -1]*Eo*Er

    sigma = [sigmaTop, sigmaRight, sigmaBottom, sigmaLeft]
    return sigma

def findCharges(sigma, k, step):
    '''
    Find the charges Qtop and Qfluid

    Input
    -----
    sigma: list of 1d numpy arrays
        charge densities (linear) on all surfaces
        Refer to findSigma() for order of surfaces
    k: int
        index corresponding to boundary
    step: float
        distance between 2 adjacent nodes

    Output
    ------
    Q: list
        [Qtop, Qfluid] charges

    '''
    # top plate charge
    QTop = np.sum(sigma[0]*step)
    print(QTop)
    # bottom surface charge
    QBottom = np.sum(sigma[2]*step)

    # left plate (submerged in dielectric) charge
    QLeftFluid = np.sum(sigma[3][:k]*step)

    # right plate (submerged in dielectric) charge
    QRightFluid = np.sum(sigma[1][:k]*step)

    # total charge in surface submerged in fluid
    QFluid = QBottom+QLeftFluid+QRightFluid
    print(QFluid)
    Q = [QTop, QFluid]
    return Q

'''
Helper functions

    plotSemilogy() - for semilogy plots
    plotQuiver() - for quiver plots
    plotContour() - for contour plots
    plot() - for linear-scale plots

'''
figNum=0                # figure number
plotsDir = 'plots/'     # plots directory
# NOTE: create plots/ directory before running the code

def plotSemilogy(x, y, figTitle=None, blockFig=False, showFig=showAll, saveFig=saveAll, xLimit=None, yLimit=None, xLabel=r"$x\ \to$", yLabel=r"$y\ \to$", multiplePlots=False, labels=None):
    '''
    Helper function to plot semilogy plots

    '''
    global figNum
    plt.figure(figNum)
    plt.title(figTitle)
    plt.grid()
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    if not multiplePlots:
        plt.semilogy(x, y)
    else:
        i=0
        for a,b in zip(x, y):
            plt.semilogy(a, b, label=labels[i])
            i=i+1
    if xLimit:
        plt.xlim(xLimit)
    if yLimit:
        plt.ylim(yLimit)
    if labels != None:
        plt.legend()
    if saveFig:
        plt.savefig(plotsDir + "Fig"+str(figNum)+".png")
    if showFig:
        plt.show(block=blockFig)
    figNum+=1

def plotContour(X, Y, f, figTitle=None, blockFig=False, showFig=showAll, saveFig=saveAll, xLimit=None, yLimit=None, xLabel=r"$x\ \to$", yLabel=r"$y\ \to$"):
    '''
    Helper function to plot Contour plots

    '''
    global figNum
    plt.figure(figNum)
    plt.title(figTitle)
    plt.grid()
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.contourf(X, Y, f)
    ax = plt.axes()
    plt.colorbar(ax=ax, orientation='vertical')
    if xLimit:
        plt.xlim(xLimit)
    if yLimit:
        plt.ylim(yLimit)
    if saveFig:
        plt.savefig(plotsDir + "Fig"+str(figNum)+".png")
    if showFig:
        plt.show(block=blockFig)
    figNum+=1

def plot(x, y, figTitle=None, style='b-', blockFig=False, showFig=showAll, saveFig=saveAll, xLimit=None, yLimit=None, xLabel=r"$x\ \to$", yLabel=r"$y\ \to$", multiplePlots=False, labels=None):
    '''
    Helper function to plot linear-scale plots

    '''
    global figNum
    plt.figure(figNum)
    plt.title(figTitle)
    plt.grid()
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    if not multiplePlots:
        plt.plot(x, y, style)
    else:
        i=0
        for a,b in zip(x, y):
            plt.plot(a, b, label=labels[i])
            i=i+1
    if xLimit:
        plt.xlim(xLimit)
    if yLimit:
        plt.ylim(yLimit)
    if labels != None:
        plt.legend()
    if saveFig:
        plt.savefig(plotsDir + "Fig"+str(figNum)+".png")
    if showFig:
        plt.show(block=blockFig)
    figNum+=1

def plotQuiver(X, Y, compX, compY, figTitle=None, blockFig=False, showFig=showAll, saveFig=saveAll, xLimit=None, yLimit=None, xLabel=r"$x\ \to$", yLabel=r"$y\ \to$"):
    '''
    Helper function plot Quiver plots

    '''
    global figNum
    plt.figure(figNum)
    plt.axes().quiver(X, Y, compX, compY)
    plt.axes().set_title(figTitle)
    plt.grid()
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    if xLimit:
        plt.xlim(xLimit)
    if yLimit:
        plt.ylim(yLimit)
    if saveFig:
        plt.savefig(plotsDir + "Fig"+str(figNum)+".png")
    if showFig:
        plt.show(block=blockFig)
    figNum+=1

def main():
    step = 1e-3
    h = 0.05*Ly
    print(h/Ly)
    partD(int(Lx/step), int(Ly/step), step, int(h/step), 1e-8, 100000)

if __name__ == "__main__":
    main()
