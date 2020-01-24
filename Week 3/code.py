'''
------------------------------------
Assignment 3 - EE2703 (Jan-May 2020)
Done by Akilesh Kannan (EE18B122)
Created on 18/01/20
Last Modified on 24/01/20
------------------------------------
'''
from pylab import *
from scipy import special as sp

def g_t(t, A, B):
    return A*sp.jn(2,t) + B*t

rawFileData = np.loadtxt("fitting.dat", usecols=(1,2,3,4,5,6,7,8,9))
fileDataColumns = [[],[],[],[],[],[],[],[],[]]
for i in range(len(rawFileData)):
    for j in range(len(rawFileData[i])):
        fileDataColumns[j].append(rawFileData[i][j])
t = linspace(0,10,101)
sigma = logspace(-1,-3,9)
sigma = np.around(sigma,3)
A = 1.05
B = -0.105
trueFunction = g_t(t, A, B)
for i in range(len(fileDataColumns)):
    plot(t,fileDataColumns[i],label='$\sigma_{} = {}$'.format(i, sigma[i]))
plot(t, trueFunction, label='true value', color='#000000')
xlabel('$t$')
ylabel('$f(t)+n(t)$')
title('Noisy plots vs True plot')
legend()
show()
