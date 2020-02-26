'''
------------------------------------
Assignment 5 - EE2703 (Jan-May 2020)
Done by Akilesh Kannan (EE18B122)
Created on 22/02/20
Last Modified on 22/02/20
------------------------------------
'''

import numpy as np
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

phi=np.zeros((Nx, Ny), dtype=float)
x=np.linspace(-radius-0.25, radius+0.25, Nx)
y=np.linspace(-radius-0.25, radius+0.25, Ny)
Y, X=np.meshgrid(-y,x)

ii = np.where(np.square(X)+np.square(Y) <= radius**2)
phi[ii] = 1.0

plt.figure(1)
contPlot = plt.contourf(X, Y, phi, cmap=cm.jet)
ax = plt.axes()
ax.set_aspect('equal')
plt.colorbar(ax=ax, orientation='vertical')
plt.title('Contour Plot of $\phi$')
plt.savefig(plotsDir+'Fig1.png')
