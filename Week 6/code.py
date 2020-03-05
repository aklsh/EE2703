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
plt.figure(1)
plt.plot(t, x, label='decay=0.5')
plt.xlabel(r"$t \to$")
plt.ylabel(r"$x(t) \to$")
plt.title('Solution of $x\'\' + 2.25x = f(t)$')
plt.legend()
plt.grid()
plt.ylim(-1, 1)
plt.show()

t, x = findX(0.05, 1.5)
plt.figure(2)
plt.plot(t, x, label='decay=0.05')
plt.xlabel(r"$t \to$")
plt.ylabel(r"$x(t) \to$")
plt.title('Solution of $x\'\' + 2.25x = f(t)$')
plt.legend()
plt.grid()
plt.show()

'''
f(t) = e^(-at)cos(wt)u(t)

t ---> time vector
decay ---> decay
w ---> frequency
'''


def ft(t, decay, w):
    return np.exp(-decay*t)*np.cos(w*t)


q3Hs = sgnl.lti([1], [1, 0, 2.25])
plt.figure(3)
for w in np.arange(1.4, 1.6, 0.05):
    tvector = np.linspace(0, 50, 1000)
    t, y, rest = sgnl.lsim(q3Hs, U=ft(tvector, 0.05, w), T=tvector)
    plt.plot(t, y, label='$w = {} rad/s$'.format(w))
    plt.legend()
plt.xlabel(r"$t \to$")
plt.ylabel(r"$x(t) \to$")
plt.grid()
plt.show()

Xs = sgnl.lti([1, 0, 2], [1, 0, 3, 0])
t, x = sgnl.impulse(Xs, None, np.linspace(0, 20, 1000))
Ys = sgnl.lti([2], [1, 0, 3, 0])
t, y = sgnl.impulse(Ys, None, np.linspace(0, 20, 1000))

plt.figure(4)
plt.plot(t, y, label=r"$y(t)$")
plt.plot(t, x, label=r"$x(t)$")
plt.xlabel(r"$t \to$")
plt.legend()
plt.show()

L = 1e-6
C = 1e-6
R = 100
q4Hs = sgnl.lti([1], [L*C, R*C, 1])
w, mag, phase = sgnl.bode(q4Hs)

plt.figure(5)
plt.semilogx(w, mag)
plt.xlabel(r"$\omega \ \to$")
plt.ylabel(r"$\|H(jw)\|$")
plt.title("Magnitude plot")
plt.show()

plt.figure(6)
plt.xlabel(r"$\omega \ \to$")
plt.ylabel(r"$\angle H(jw)$")
plt.title("Phase plot")
plt.semilogx(w, phase)
plt.show()

