'''
------------------------------------
Assignment 5 - EE2703 (Jan-May 2020)
Done by Akilesh Kannan (EE18B122)
Created on 22/02/20
Last Modified on 22/02/20
------------------------------------
'''

import numpy as np
import scipy.linalg
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import argparse
import sys

plotsDir = 'plots/'

parser = argparse.ArgumentParser()
parser.add_argument("--Nx", help='size along x', type=int, default=25, metavar='nx')
parser.add_argument("--Ny", help='size along y', type=int, default=25, metavar='ny')
parser.add_argument("--radius", help='radius of central lead', type=float, default=8, metavar='r')
parser.add_argument("--Niter", help='number of iterations to perform', type=int, default=1500, metavar='ni')
args=parser.parse_args()
[Nx, Ny, radius, Niter] = [args.Nx, args.Ny, args.radius, args.Niter]

phi = np.zeros((Nx, Ny), dtype=float)
x = np.linspace(-radius*1.25, radius*1.25, Nx)
y = np.linspace(-radius*1.25, radius*1.25, Ny)
Y, X = np.meshgrid(-y,x)

volt1Nodes = np.where(np.square(X)+np.square(Y) <= radius**2)
phi[volt1Nodes] = 1.0

plt.figure(1)
contPlot = plt.contourf(X, Y, phi, cmap=cm.jet)
ax = plt.axes()
ax.set_aspect('equal')
plt.colorbar(ax=ax, orientation='vertical')
plt.title('Contour Plot of $\phi$')
plt.savefig(plotsDir+'Fig1.png')
plt.show(block=False)

# Start iterations
iteration=[]
error=[]
for x in range(Niter):
    # Copy old phi
    oldphi = phi.copy()

    # Updating the Potential
    phi[1:-1, 1:-1] = 0.25*(phi[1:-1, 0:-2]+phi[1:-1, 2:]+phi[0:-2, 1:-1]+phi[2:, 1:-1])

    # Applying Boundary Conditions
    phi[1:-1, 0] = phi[1:-1, 1]  # Left edge
    phi[1:-1, -1] = phi[1:-1, -2]  # right edge
    phi[0, :] = phi[1, :]  # Top edge
    # Bottom edge is grounded so no boundary conditions

    # Assigning 1V to electrode region
    phi[volt1Nodes] = 1.0

    error.append(np.abs(phi-oldphi).max())
    iteration.append(x)

plt.figure(2)
plt.semilogy(iteration, error)
ax = plt.axes()
ax.set_aspect('equal')
plt.title('Semilog Plot of error')
plt.xlabel('iteration')
plt.ylabel('error')
plt.savefig(plotsDir+'Fig2.png')
plt.show(block=False)

plt.figure(3)
plt.loglog(iteration, error)
ax = plt.axes()
plt.title('Loglog Plot of error')
plt.xlabel('iteration')
plt.ylabel('error')
plt.savefig(plotsDir+'Fig3.png')
plt.show(block=False)

def findFit(errors, iterations):
    nRows = len(errors)
    coeffMatrix = np.zeros((nRows,2), dtype=float)
    constMatrix = np.zeros((nRows,1), dtype=float)
    coeffMatrix[:,0] = 1
    coeffMatrix[:,1] = iterations
    constMatrix = np.log(errors)
    fit = scipy.linalg.lstsq(coeffMatrix, constMatrix)[0]
    estimate = coeffMatrix@fit
    return fit, estimate

fitAll, estimateAll = findFit(error, iteration)
fitAfter500, estimateAfter500 = findFit(error[501:], iteration[501:])

plt.figure(4)
plt.semilogy(iteration, np.exp(estimateAll), 'r.', mfc='none', label='fit all')
plt.semilogy(iteration[501:], np.exp(estimateAfter500), 'y.', mfc='none', label='fit after 500')
plt.semilogy(iteration, error, 'g', label='actual error')
plt.title('Comparison of actual errors and fits')
plt.xlabel('iteration')
plt.ylabel('error / fit')
plt.legend()
plt.savefig(plotsDir+'Fig4.png')
plt.show(block=True)

def cummulError(N, A, B):
    return -(A/B)*np.exp(B*(N+0.5))

def findStopCond(errors, Niter, error_tol):
    cummulErr = []
    for n in range(1, Niter):
        cummulErr.append(cummulError(n, np.exp(fitAll[0]), fitAll[1]))
        if(cummulErr[n-1] <= error_tol):
            print("last per-iteration change in the error is %g" % (np.abs(cummulErr[-1]-cummulErr[-2])))
            return cummulErr[n-1], n
    print("last per-iteration change in the error is %g" % (np.abs(cummulErr[-1]-cummulErr[-2])))
    return cummulErr[-1], Niter

errorTol = 10e-8
cummulErr, nStop = findStopCond(error, Niter, errorTol)
print("Stopping Condition N: %g and Error is %g" % (nStop, cummulErr))
