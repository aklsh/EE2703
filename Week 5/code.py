'''
------------------------------------
Assignment 5 - EE2703 (Jan-May 2020)
Done by Akilesh Kannan (EE18B122)
Created on 22/02/20
Last Modified on 02/03/20
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
X, Y = np.meshgrid(x, -y)

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
iteration = []
error = []
for n in range(Niter):
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
    iteration.append(n)

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
plt.semilogy(iteration[::200], np.exp(estimateAll[::200]), 'r.', mfc='none', label='fit all')
plt.semilogy(iteration[501::200], np.exp(estimateAfter500[::200]), 'y.', mfc='none', label='fit after 500')
plt.semilogy(iteration, error, 'g', label='actual error')
plt.title('Comparison of actual errors and fits')
plt.xlabel('iteration')
plt.ylabel('error / fit')
plt.legend()
plt.savefig(plotsDir+'Fig4.png')
plt.show(block=False)

def cummulError(N, A, B):
    return -(A/B)*np.exp(B*(N+0.5))

def findStopCond(errors, Niter, error_tol):
    cummulErr = []
    for n in range(1, Niter):
        cummulErr.append(cummulError(n, np.exp(fitAll[0]), fitAll[1]))
        if(cummulErr[n-1] <= error_tol):
            print("last \033[1mper-iteration change\033[0;0m in the error is ", (np.abs(cummulErr[-1]-cummulErr[-2])))
            return cummulErr[n-1], n
    print("last \033[1mper-iteration change\033[0;0m in the error is ", (np.abs(cummulErr[-1]-cummulErr[-2])))
    return cummulErr[-1], Niter

errorTol = 10e-8
cummulErr, nStop = findStopCond(error, Niter, errorTol)
print("\033[1mStopping Condition\033[0;0m ----> N: %g and Error: %g" % (nStop, cummulErr))

fig5 = plt.figure(5)
ax = p3.Axes3D(fig5)
plt.title('The 3-D surface plot of $\phi$')
surfacePlot = ax.plot_surface(X, Y, phi, rstride=1, cstride=1, cmap=cm.jet)
cax = fig5.add_axes([1, 0, 0.1, 1])
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
fig5.colorbar(surfacePlot, cax=cax, orientation='vertical')
plt.show(block=False)
plt.savefig(plotsDir+'Fig5.png')

plt.figure(6)
plt.contourf(X, Y, phi, cmap=cm.jet)
ax = plt.axes()
ax.set_aspect('equal')
plt.colorbar(ax=ax, orientation='vertical')
plt.title('Updated Contour Plot of $\phi$')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(plotsDir+'Fig6.png')
plt.show(block=False)

Jx = np.zeros((Ny,Nx))
Jy = np.zeros((Ny,Nx))

Jx[1:-1, 1:-1] = (phi[1:-1, 0:-2] - phi[1:-1, 2:])/2.0
Jx[1:-1, 1:-1] = (phi[0:-2, 1:-1] - phi[2:, 1:-1])/2.0

plt.figure(7)
plt.scatter(x[volt1Nodes[0]], y[volt1Nodes[1]], color='r', s=12, label='$V = 1V$ region')
plt.axes().quiver(y, x, Jy[::-1,:], Jx[::-1,:])
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show(block=True)
plt.title('Vector plot of flow of current')
plt.savefig(plotsDir+'Fig7.png')
plt.show(block=True)
