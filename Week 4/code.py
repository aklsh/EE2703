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
grid()
show()
figure(2)
ax2 = axes()
ax2.xaxis.set_major_formatter(FuncFormatter(pi_tick))
semilogy(x, exFunc(x), 'k', label='True Function', xunits=radians)
semilogy(x, exFunc(x%(2*pi)), '--', label='Fourier Series expansion', xunits=radians)
legend(loc='upper left')
grid()
show()

def cosCoeff(x, k, f):
    return f(x)*cos(k*x)
def sinCoeff(x, k, f):
    return f(x)*sin(k*x)

coscosA = []
coscosB = []
expA = []
expB = []
coscosB.append(0)
expB.append(0)
coscosA.append(quad(coscosxFunc, 0, 2*pi)[0]/(2*pi))
expA.append(quad(exFunc, 0, 2*pi)[0]/(2*pi))
for i in range(1, 26):
    coscosA.append((quad(cosCoeff, 0, 2*pi, args=(i, coscosxFunc))[0])/pi)
    expA.append((quad(cosCoeff, 0, 2*pi, args=(i, exFunc))[0])/pi)
    coscosB.append((quad(sinCoeff, 0, 2*pi, args=(i, coscosxFunc))[0])/pi)
    expB.append((quad(sinCoeff, 0, 2*pi, args=(i, exFunc))[0])/pi)
# print(coscosA)
# print(coscosB)
cosSeriesCosCos = sinSeriesCosCos = cosSeriesExp = sinSeriesExp = [0]*400
for i in range(400):
    for j in range(26):
        cosSeriesCosCos[i]+=(coscosA[j]*cos(j*x[i]))
        sinSeriesCosCos[i]+=(coscosB[j]*sin(j*x[i]))
        cosSeriesExp[i]+=(expA[j]*cos(j*x[i]))
        sinSeriesExp[i]+=(expB[j]*sin(j*x[i]))
print(cosSeriesCosCos)
fourierCosCos = [cosSeriesCosCos[i] + sinSeriesCosCos[i] for i in range(400)]
fourierExp = [cosSeriesExp[i] + sinSeriesExp[i] for i in range(400)]
figure(3)
ax3 = axes()
ax3.xaxis.set_major_formatter(FuncFormatter(pi_tick))
plot(x, fourierCosCos, 'ro', label='FS of $cos(cos(x))$', xunits=radians)
legend()
show()
figure(4)
ax4 = axes()
ax4.xaxis.set_major_formatter(FuncFormatter(pi_tick))
semilogy(x, fourierExp, 'ro', label='FS of $e^x$', xunits=radians)
ylim([pow(10, -1), pow(10, 4)])
legend()
show()
