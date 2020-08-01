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
import matplotlib
matplotlib.rcParams['axes.formatter.useoffset'] = False

# Define constants in SI units, wherever applicable
Lx = 0.1                # width of tank
Ly = 0.2                # height of tank
Eo = 8.85e-12           # permittivity of free space
Er = 2                  # dielectric constant of water

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

def solve(M, N, step, k, accuracy, No, plotAll=False):
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
    plotAll: bool
        switch to plot data
        True - plot data
        False - no plotting

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

    if plotAll:
        plotContour(X, Y, phi, figTitle='Initial potential distribution')

    iteration=[]     # iteration number
    error=[]         # error vector

    # iteratively calculate potentials
    for i in range(No):
        # create copy of potentials
        oldPhi = phi.copy()

        # updating the potentials
        phi[1:-1, 1:-1] = 0.25*(phi[1:-1, 0:-2]+phi[1:-1, 2:]+phi[0:-2, 1:-1]+phi[2:, 1:-1])
        phi[k, 1:-1] = (Er*oldPhi[k-1, 1:-1] + oldPhi[k+1, 1:-1])*1.0/(1+Er)

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

    if plotAll:
        plotContour(X, Y, phi, figTitle='Potential distribution after updating')

        # find LSTSQ Estimate for exponential region (>5000 iterations)
        fit, estimate = findExpFit(error[5000:], iteration[5000:], printFit=True)

        # extrapolate the estimated error function till iteration 0
        estimate = np.e**(fit[0]+np.multiply(fit[1], iteration))

        plotSemilogy([iteration, iteration], [error, estimate], multiplePlots=True, labels=["Actual error", "Fitted error (iteration >= 5000)"], figTitle='Error vs. iteration', xLabel=r"iterations $\to$", yLabel=r'error $\to$')

    # calculate E
    Ex, Ey = findEField(phi, step, M, N, plotAll)
    checkContinuity(Ex, Ey, k, M, plotAll)

    # calculate charge densities
    sigma = findSigma(Ex, Ey, k)

    # calculate charges Qtop and Qfluid
    Q = findCharges(sigma, k, step)

    # calculate angles with normal
    angleBelow = findAngles(Ex[k-1, :], Ey[k-1, :])
    angleAbove = findAngles(Ex[k, :], Ey[k, :])

    if plotAll:
        x = np.linspace(0, Lx, M-1, dtype=float)
        sineAnglesBelow = np.sin(angleBelow)
        sineAnglesAbove = np.sin(angleAbove)
        tanAnglesBelow = np.tan(angleBelow)
        tanAnglesAbove = np.tan(angleAbove)
        plot(x, np.divide(sineAnglesBelow, sineAnglesAbove), r"Ratio of sine of angle with normal above and below", yLabel=r"$\frac{sin\,\theta_a}{sin\,\theta_b}$")
        plot(x, np.divide(tanAnglesBelow, tanAnglesAbove), r"Ratio of tangent of angle with normal above and below", yLabel=r"$\frac{tan\,\theta_a}{tan\,\theta_b}$")

    return phi, Q, iteration[-1], error

def findEField(phi, step, M, N, plotAll):
    '''
    Calculates the x- and y- components of E-field at
    each point.

    Input
    -----
    phi: 2d numpy array
        potential array
    step: float
        distance between 2 points on the grid
    M: int
        nodes along x-axis
    N: int
        nodes along y-axis
    plotAll: bool
        switch to plot data
        True - plot data
        False - no plotting

    Output
    ------
    Ex: 2d numpy array
        X-components of E field
    Ey: 2d numpy array
        Y-components of E-field

    '''
    # Ex calculation
    #   *   *   *     row i
    #     -   -     --> center of mesh cells
    #   *   *   *     row i+1
    #
    negativeGradientX = (phi[:, :-1] - phi[:, 1:])*(1.0/step)
    Ex = (negativeGradientX[:-1, :] + negativeGradientX[1:, :])*0.5

    # Ey calculation
    #   *       *
    #       -       --> center of mesh cells
    #   *       *
    #       -       --> center of mesh cells
    #   *       *
    # col i   col i+1
    #
    negativeGradientY = (phi[:-1, :] - phi[1:, :])*(1.0/step)
    Ey = (negativeGradientY[:, :-1] + negativeGradientY[:, 1:])*0.5

    # plot
    if plotAll:
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

    # bottom surface charge
    QBottom = np.sum(sigma[2]*step)

    # left plate (submerged in dielectric) charge
    QLeftFluid = np.sum(sigma[3][:k]*step)

    # right plate (submerged in dielectric) charge
    QRightFluid = np.sum(sigma[1][:k]*step)

    # total charge in surface submerged in fluid
    QFluid = QBottom+QLeftFluid+QRightFluid

    Q = [QTop, QFluid]
    return Q

def findAngles(Ex, Ey):
    '''
    Find the angle b/w y-axis and E-field at all
    points on the grid

    Input
    -----
    Ex: 2d numpy array
        X-component of E-field
    Ey: 2d numpy array
        Y-component of E-field

    Output
    ------
    angle: 2d numpy array
        angle b/w E-field and y-axis at all points
        on the grid

    '''

    # angle = atan(Ex/Ey)
    ## NOTE: angle is calculated wrt y-axis
    angles = np.arctan2(Ex, Ey)
    return angles

def checkContinuity(Ex, Ey, k, M, plotAll):
    '''
    Function to verify continuity of Dn and
    Et across interface

    Input
    -----
    Ex: 2d numpy array
        X-component of E-field
    Ey: 2d numpy array
        Y-component of E-field
    k: int
        index corresponding to height of fluid
    M: int
        number of nodes across x-axis
    plotAll: bool
        switch to plot data
        True - plot data
        False - no plotting

    '''
    if plotAll:
        x = np.linspace(0, Lx, M-1)
        # checking Dn continuity
        plot([x, x], [Ey[k-1, :]*Er, Ey[k, :]], multiplePlots=True, labels=["Below boundary", "Above boundary"], yLabel=r"$D_{normal}$", figTitle=r"Continuity of $D_{normal}$ across boundary")

        # checking Et continuity
        plot([x, x], [Ex[k-1, :], Ex[k, :]], multiplePlots=True, labels=["Below boundary", "Above boundary"], yLabel=r"$E_{tangential}$", figTitle=r"Continuity of $E_{tangential}$ across boundary")

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

saveAll = True
showAll = False

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
        style=["+", "o"]
        for a,b in zip(x, y):
            plt.plot(a, b, style[i], label=labels[i])
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
    accuracy = 1e-8
    maxIter = 100000
    h = np.linspace(0, 1, 10, dtype=float, endpoint=False)
    k =h*(Ly/step)
    Q = [None]*10
    phi = [None]*10
    for x in range(10):
        phi[x], Q[x], *args = solve(int(Lx/step), int(Ly/step), step, int(k[x]), accuracy, maxIter, x==5)
    QTop = [x[0] for x in Q]
    QFluid =[x[1] for x in Q]

    plot(h, QTop, r"$Q_{top}$ vs. h/$L_y$", yLabel=r"$Q_{top}$ in pC", xLabel=r"$h/L_y$")
    plot(h, QFluid, r"$Q_{fluid}$ vs. h/$L_y$", yLabel="$Q_{fluid}$ in pC", xLabel="$h/L_y$")

if __name__ == "__main__":
    main()
