'''
------------------------------------
Assignment 6 - EE2703 (Jan-May 2020)
Done by Akilesh Kannan (EE18B122)
Created on 04/03/20
Last Modified on 04/03/20
------------------------------------
'''

import scipy.signal as sgnl
import numpy as np
import matplotlib.pyplot as plt

'''
f(t) = e^(-at)cos(wt)u(t)
a ---> decay
w ---> frequency
'''


def findX(a, w):
    XsNum = np.poly1d([1, a])
    XsDen = np.polymul([1, 2*a, a**2 + 2.25], [1, 0, 2.25])
    Xs = sgnl.lti(XsNum, XsDen)
    t, x = sgnl.impulse(Xs, None, np.linspace(0, 50, 1000))
    return t, x


t, x = findX(0.5, 1.5)
plt.plot(t, x, label='decay=0.5')
plt.xlabel(r"$t \to$")
plt.ylabel(r"$x(t) \to$")
plt.title('Solution of $x\'\' + 2.25x = f(t)$')
plt.legend()
plt.grid()
plt.ylim(-1, 1)
plt.show()

t, x = findX(0.05, 1.5)
plt.plot(t, x, label='decay=0.05')
plt.xlabel(r"$t \to$")
plt.ylabel(r"$x(t) \to$")
plt.title('Solution of $x\'\' + 2.25x = f(t)$')
plt.legend()
plt.grid()
plt.show()


def ft(t, decay, w):
    return np.exp(-decay*t)*np.cos(w*t)


q3Hs = sgnl.lti([1], [1, 0, 2.25])
for w in np.arange(1.4, 1.6, 0.05):
    t, y, rest = sgnl.lsim(q3Hs, U=ft(np.linspace(0, 50, 1000), 0.05, w), T=np.linspace(0, 50, 1000))
    plt.plot(t, y, label='$w = {} rad/s$'.format(w))
    plt.legend()
plt.xlabel(r"$t \to$")
plt.ylabel(r"$x(t) \to$")
plt.grid()
plt.show()
