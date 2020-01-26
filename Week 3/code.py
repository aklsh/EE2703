'''
------------------------------------
Assignment 3 - EE2703 (Jan-May 2020)
Done by Akilesh Kannan (EE18B122)
Created on 18/01/20
Last Modified on 26/01/20
------------------------------------
'''

# Importing necessary libraries
from pylab import *
from scipy import special as sp
import os

# QUESTION: 2
try:
    rawFileData = np.loadtxt("fitting.dat", usecols=(1,2,3,4,5,6,7,8,9))
except OSError:
    sys.exit("fitting.dat not found! Please run the code in generateData.py before you run this code.")
fileDataColumns = [[],[],[],[],[],[],[],[],[]]
for i in range(len(rawFileData)):
    for j in range(len(rawFileData[i])):
        fileDataColumns[j].append(rawFileData[i][j])

# QUESTION: 3
t = linspace(0,10,101)
sigma = logspace(-1,-3,9)
sigma = around(sigma,3)

figure(0)
for i in range(len(fileDataColumns)):
    plot(t,fileDataColumns[i],label='$\sigma_{} = {}$'.format(i, sigma[i]))

# QUESTION: 4
def g_t(t, A, B):
    return A*sp.jn(2,t) + B*t
A = 1.05
B = -0.105
trueFunction = g_t(t, A, B)
plot(t, trueFunction, label='true value', color='#000000')
xlabel('$t$')
ylabel('$f(t)+n(t)$')
title('Noisy plots vs True plot')
legend()
show()

# QUESTION: 5
figure(1)
xlabel('$t$')
ylabel('$f(t)$')
title('Errorbar Plot')
plot(t, trueFunction, label='f(t)', color='#000000')
errorbar(t[::5],fileDataColumns[0][::5],0.1, fmt='bo', label='Error Bar')
legend()
show()

# QUESTION: 6
jColumn = sp.jn(2,t)
M = c_[jColumn, t]
p = array([A, B])
actual = c_[t,trueFunction]
print('Question 6 - The 2 vectors are equal') if((M@p == trueFunction).all()) else print('Questio 6 - The 2 vectors are not equal')

# QUESTION: 7
A = arange(0,2,0.1)
B = arange(-0.2,0,0.01)
epsilon = zeros((len(A), len(B)))
for i in range(len(A)):
    for j in range(len(B)):
            epsilon[i][j] = mean(square(fileDataColumns[0][:] - g_t(t[:], A[i], B[j])))

# QUESTION: 8
figure(2)
contPlot=contour(A,B,epsilon,levels=20)
xlabel("A")
ylabel("B")
title("Contours of $\epsilon_{ij}$")
clabel(contPlot, inline=1, fontsize=10)
plot([1.05], [-0.105], 'ro')
grid()
annotate("Exact Location\nof Minima", (1.05, -0.105), xytext=(-50, -40), textcoords="offset points", arrowprops={"arrowstyle": "->"})
show()

# QUESTION: 9
p, *rest = lstsq(M,trueFunction,rcond=None)

# QUESTION: 10
figure(3)
perr=zeros((9, 2))
for k in range(len(fileDataColumns)):
    perr[k], *rest = lstsq(M, fileDataColumns[k], rcond=None)
Aerr = array([square(x[0]-p[0]) for x in perr])
Berr = array([square(x[1]-p[1]) for x in perr])
plot(sigma, Aerr, '+--', label='$A_{err}$')
plot(sigma, Berr, '+--', label='$B_{err}$')
xlabel("$\sigma_{noise}$")
title("Variation of error with noise")
ylabel("MSerror")
legend()
grid()
show()
