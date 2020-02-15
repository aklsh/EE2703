'''
------------------------------------
Assignment 4 - EE2703 (Jan-May 2020)
Done by Akilesh Kannan (EE18B122)
Created on 09/02/20
Last Modified on 13/02/20
------------------------------------
'''

import numpy as np
from scipy.integrate import quad
from matplotlib import pyplot as plt

def coscosxFunc(x):
    return np.cos(np.cos(x))
def expFunc(x):
    return np.exp(x)

def pi_tick(value, tick_number):
    # find number of multiples of pi/2
    N = int(round(2 * value / PI))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\pi/2$"
    elif N == -1:
        return r"$-\pi/2$"
    elif N == 2:
        return r"$\pi$"
    elif N == -2:
        return r"$-\pi$"
    elif N % 2 != 0:
        return r"${0}\pi/2$".format(N)
    else:
        return r"${0}\pi$".format(N//2)

PI = np.pi
plotsDir = 'plots/'
x = np.linspace(-2*PI, 4*PI, 1201)
x = x[:-1]

plt.figure('Figure 1')
ax1 = plt.axes()
ax1.xaxis.set_major_formatter(plt.FuncFormatter(pi_tick))
plt.plot(x, coscosxFunc(x), 'k', label='True Function')
plt.plot(x, coscosxFunc(x%(2*PI)), '--', label='Fourier Series expansion')
plt.axis([-6, 10, 0.5, 1.05])
plt.legend(loc='upper right')
plt.title('$cos(cos(x))$')
plt.grid()
plt.savefig(plotsDir+'Figure 1.png')

plt.figure('Figure 2')
ax2 = plt.axes()
ax2.xaxis.set_major_formatter(plt.FuncFormatter(pi_tick))
plt.semilogy(x, expFunc(x), 'k', label='True Function')
plt.semilogy(x, expFunc(x%(2*PI)), '--', label='Fourier Series expansion')
plt.legend(loc='upper left')
plt.title('$e^x$')
plt.grid()
plt.savefig(plotsDir+'Figure 2.png')

def cosCoeff(x, k, f):
    return f(x)*np.cos(k*x)
def sinCoeff(x, k, f):
    return f(x)*np.sin(k*x)

def calc51FourierCoeffs (f):
    aCoeff = np.zeros(26)
    bCoeff = np.zeros(26)
    aCoeff[0] = quad(cosCoeff, 0, 2*PI, args=(0, f))[0]/(2*PI)
    for i in range(1, 26):
        aCoeff[i] = quad(cosCoeff, 0, 2*PI, args=(i, f))[0]/(PI)
        bCoeff[i] = quad(sinCoeff, 0, 2*PI, args=(i, f))[0]/(PI)
    coeffs = np.zeros(51)
    coeffs[0] = aCoeff[0]
    coeffs[1::2] = aCoeff[1:]
    coeffs[2::2] = bCoeff[1:]
    return coeffs

coeffCosCos = calc51FourierCoeffs(coscosxFunc)
coeffExp = calc51FourierCoeffs(expFunc)

xTicksForCoeffsSemilog = ['$a_0$']
for i in range(1, 26):
    xTicksForCoeffsSemilog.append('$a_{'+str(i)+'}$')
    xTicksForCoeffsSemilog.append('$b_{'+str(i)+'}$')

plt.figure('Figure 3')
plt.xticks(np.arange(51), xTicksForCoeffsSemilog, rotation=60)
plt.tick_params(axis='x', labelsize=7)
plt.semilogy(abs(coeffCosCos[1::2]), 'bo', label='$a_n$')
plt.semilogy(abs(coeffCosCos[2::2]), 'yo', label='$b_n$')
plt.legend()
plt.title('$cos(cos(x))$ semilog plot')
plt.grid()
plt.savefig(plotsDir+'Figure 3.png')

plt.figure('Figure 4')
plt.xticks(np.arange(51), xTicksForCoeffsSemilog, rotation=60)
plt.tick_params(axis='x', labelsize=7)
plt.semilogy(abs(coeffExp[1::2]), 'bo', label='$a_n$')
plt.semilogy(abs(coeffExp[2::2]), 'yo', label='$b_n$')
plt.legend()
plt.title('$e^x$ semilog plot')
plt.grid()
plt.savefig(plotsDir+'Figure 4.png')

plt.figure('Figure 5')
plt.loglog(abs(coeffCosCos[1::2]), 'bo', label='$a_n$')
plt.loglog(abs(coeffCosCos[2::2]), 'yo', label='$b_n$')
plt.legend()
plt.title('$cos(cos(x))$ loglog plot')
plt.grid()
plt.savefig(plotsDir+'Figure 5.png')

plt.figure('Figure 6')
plt.loglog(abs(coeffExp[1::2]), 'bo', label='$a_n$')
plt.loglog(abs(coeffExp[2::2]), 'yo', label='$b_n$')
plt.legend()
plt.title('$e^x$ loglog plot')
plt.grid()
plt.savefig(plotsDir+'Figure 6.png')

def findLSTSQCoeff(f):
    x = np.linspace(0, 2*PI, 401)
    x = x[:-1]
    b = f(x)
    M = np.zeros((400, 51))
    M[:,0] = 1
    for k in range(1,26):
        M[:,(2*k)-1]=np.cos(k*x)
        M[:,2*k]=np.sin(k*x)
    return np.linalg.lstsq(M, b, rcond=None)[0]

lstsqCosCos = findLSTSQCoeff(coscosxFunc)
lstsqExp = findLSTSQCoeff(expFunc)

plt.figure('Figure 3.1')
plt.semilogy(abs(coeffCosCos[1::2]), 'ro', label='$a_n$ by Integration')
plt.semilogy(abs(coeffCosCos[2::2]), 'bo', label='$b_n$ by Integration')
plt.semilogy(abs(lstsqCosCos[1::2]), 'go', label='$a_n$ by lstsq')
plt.semilogy(abs(lstsqCosCos[2::2]), 'yo', label='$b_n$ by lstsq')
plt.title('Comparing $cos(cos(x))$ FS coefficients - semilogy plot')
plt.legend()
plt.grid()
plt.savefig(plotsDir+'Figure 3.1.png')

plt.figure('Figure 4.1')
plt.semilogy(abs(coeffExp[1::2]), 'ro', label='$a_n$ by Integration')
plt.semilogy(abs(coeffExp[2::2]), 'bo', label='$b_n$ by Integration')
plt.semilogy(abs(lstsqExp[1::2]), 'go', label='$a_n$ by lstsq')
plt.semilogy(abs(lstsqExp[2::2]), 'yo', label='$b_n$ by lstsq')
plt.title('Comparing $e^x$ FS coefficients - semilogy plot')
plt.legend()
plt.grid()
plt.savefig(plotsDir+'Figure 4.1.png')

plt.figure('Figure 5.1')
plt.loglog(abs(coeffCosCos[1::2]), 'ro', label='$a_n$ by Integration')
plt.loglog(abs(coeffCosCos[2::2]), 'bo', label='$b_n$ by Integration')
plt.loglog(abs(lstsqCosCos[1::2]), 'go', label='$a_n$ by lstsq')
plt.loglog(abs(lstsqCosCos[2::2]), 'yo', label='$b_n$ by lstsq')
plt.title('Comparing $cos(cos(x))$ FS coefficients - loglog plot')
plt.legend()
plt.grid()
plt.savefig(plotsDir+'Figure 5.1.png')

plt.figure('Figure 6.1')
plt.loglog(abs(coeffExp[1::2]), 'ro', label='$a_n$ by Integration')
plt.loglog(abs(coeffExp[2::2]), 'bo', label='$b_n$ by Integration')
plt.loglog(abs(lstsqExp[1::2]), 'go', label='$a_n$ by lstsq')
plt.loglog(abs(lstsqExp[2::2]), 'yo', label='$b_n$ by lstsq')
plt.title('Comparing $e^x$ FS coefficients - loglog plot')
plt.legend()
plt.grid()
plt.savefig(plotsDir+'Figure 6.1.png')

absErrorsCosCos = [abs(coeffCosCos[i]-lstsqCosCos[i]) for i in range(len(coeffCosCos))]
absErrorsExp = [abs(coeffExp[i]-lstsqExp[i]) for i in range(len(coeffExp))]

print('Max. deviation for cos(cos(x)): '+str(max(absErrorsCosCos)))
print('Max. deviation for exp(x): '+str(max(absErrorsExp)))

xNew = np.linspace(-2*PI, 4*PI, 1201)
xNew = xNew[:-1]

matrixA = np.zeros((1200,51))
matrixA[:,0] = 1
for k in range(1,26):
    matrixA[:,(2*k)-1]=np.cos(k*xNew)
    matrixA[:,2*k]=np.sin(k*xNew)

plt.figure('Figure 7')
ax3 = plt.axes()
ax3.xaxis.set_major_formatter(plt.FuncFormatter(pi_tick))
plt.plot(xNew, matrixA@coeffCosCos, 'go', label='Integration')
plt.plot(xNew, matrixA@lstsqCosCos, 'r', label='lstsq')
plt.title('Reconstruction of $cos(cos(x))$')
plt.legend()
plt.grid()
plt.savefig(plotsDir+'Figure 7.png')

plt.figure('Figure 8')
ax4 = plt.axes()
ax4.xaxis.set_major_formatter(plt.FuncFormatter(pi_tick))
plt.plot(xNew, matrixA@coeffExp, 'go', label='Integration')
plt.plot(xNew, matrixA@lstsqExp, 'r', label='lstsq')
plt.title('Reconstruction of $e^x$')
plt.legend()
plt.grid()
plt.savefig(plotsDir+'Figure 8.png')
