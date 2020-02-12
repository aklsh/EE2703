'''
------------------------------------
Assignment 4 - EE2703 (Jan-May 2020)
Done by Akilesh Kannan (EE18B122)
CrexpAted on 09/02/20
Last Modified on 09/02/20
------------------------------------
'''

from pylab import *
from scipy.integrate import quad
import sys

def coscosxFunc(x):
    return cos(cos(x))

def exFunc(x):
    return exp(x)

def pi_tick(value, tick_number):
    # find number of multiples of pi/2
    N = int(round(2 * value / pi))
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

x = linspace(-2*pi, 4*pi, 400)

figure(1)
ax1 = axes()
ax1.xaxis.set_major_formatter(FuncFormatter(pi_tick))
plot(x, coscosxFunc(x), 'k', label='True Function', xunits=radians)
plot(x, coscosxFunc(x%(2*pi)), '--', label='Fourier Series expansion', xunits=radians)
axis([-6, 10, -1, 2])
legend(loc='upper right')
title('$cos(cos(x))$')
grid()
show()

figure(2)
ax2 = axes()
ax2.xaxis.set_major_formatter(FuncFormatter(pi_tick))
semilogy(x, exp(x), 'k', label='True Function', xunits=radians)
semilogy(x, exp(x%(2*pi)), '--', label='Fourier Series expansion', xunits=radians)
legend(loc='upper left')
title('$e^x$')
grid()
show()

def cosCoeff(x, k, f):
    return f(x)*cos(k*x)
def sinCoeff(x, k, f):
    return f(x)*sin(k*x)

def calc51FourierCoeffs (f):
    aCoeff = np.zeros(26)
    bCoeff = np.zeros(25)
    aCoeff[0] = quad(cosCoeff, 0, 2*pi, args=(0, f))[0]/(2*pi)
    for i in range(1, 26):
        aCoeff[i] = quad(cosCoeff, 0, 2*pi, args=(i, f))[0]/(pi)
        bCoeff[i-1] = quad(sinCoeff, 0, 2*pi, args=(i+1, f))[0]/(pi)
    coeffs = np.zeros(51)
    coeffs[0] = aCoeff[0]
    coeffs[1::2] = aCoeff[1:]
    coeffs[2::2] = bCoeff
    return coeffs

coeffCosCos = calc51FourierCoeffs(coscosxFunc)
coeffExp = calc51FourierCoeffs(exp)

xTicksForCoeffsSemilog = ['$a_0$']
for i in range(1, 26):
    xTicksForCoeffsSemilog.append('$a_{'+str(i)+'}$')
    xTicksForCoeffsSemilog.append('$b_{'+str(i)+'}$')

figure(3)
xticks(np.arange(51), xTicksForCoeffsSemilog, rotation=60)
tick_params(axis='x', labelsize=7)
semilogy(abs(coeffCosCos), 'ro')
title('$cos(cos(x))$ semilog plot')
grid()
show()

figure(4)
xticks(np.arange(51), xTicksForCoeffsSemilog, rotation=60)
tick_params(axis='x', labelsize=7)
semilogy(abs(coeffExp), 'ro')
title('$e^x$ semilog plot')
grid()
show()

xTicksForCoeffsLogLog = [0]
for i in range(1, 26):
    xTicksForCoeffsLogLog.append(i)
    xTicksForCoeffsLogLog.append(i)

figure(5)
xticks(np.arange(51), xTicksForCoeffsLogLog, rotation=60)
tick_params(axis='x', labelsize=7)
loglog(abs(coeffCosCos), 'ro')
title('$cos(cos(x))$ loglog plot')
grid()
show()

figure(6)
xticks(np.arange(51), xTicksForCoeffsLogLog, rotation=60)
tick_params(axis='x', labelsize=7)
loglog(abs(coeffExp), 'ro')
title('$e^x$ loglog plot')
grid()
show()

def createMatrix400by51(x):
    M = zeros((400, 51), dtype=float, order='C')
    M[:][0] = 1
    for k in range(1,26):
        M[:,2*k-1]=cos(k*x) # cos(kx) column
        M[:,2*k]=sin(k*x)   # sin(kx) column
    return M
matrixA = createMatrix400by51(x)
matrixB = coscosxFunc(x)
matrixC, *rest = lstsq(matrixA, matrixB)
print(matrixC)
print('')
print(coeffCosCos)
