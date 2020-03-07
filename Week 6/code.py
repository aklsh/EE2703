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

plotsDir = 'plots/'

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
plt.title(r'Solution of $\ddot{x} + 2.25x = e^{-0.5t}cos(1.5t)u(t)$')
plt.legend()
plt.grid()
plt.ylim(-1, 1)
plt.savefig(plotsDir+"Fig 1.png")

t, x = findX(0.05, 1.5)
plt.figure(2)
plt.plot(t, x, label='decay=0.05')
plt.xlabel(r"$t \to$")
plt.ylabel(r"$x(t) \to$")
plt.title(r'Solution of $\ddot{x} + 2.25x = e^{-0.05t}cos(1.5t)u(t)$')
plt.legend()
plt.grid()
plt.savefig(plotsDir+"Fig 2.png")

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
plt.title(r"Response of LTI system to various frequencies")
plt.grid()
plt.savefig(plotsDir+"Fig 3.png")

Xs = sgnl.lti([1, 0, 2], [1, 0, 3, 0])
t, x = sgnl.impulse(Xs, None, np.linspace(0, 20, 1000))
Ys = sgnl.lti([2], [1, 0, 3, 0])
t, y = sgnl.impulse(Ys, None, np.linspace(0, 20, 1000))

plt.figure(4)
plt.plot(t, y, label=r"$y(t)$")
plt.plot(t, x, label=r"$x(t)$")
plt.xlabel(r"$t \to$")
plt.title(r"$\ddot{x}+(x-y)=0$" "\n" r"$\ddot{y}+2(y-x)=0$" "\n" r"ICs: $x(0)=1,\ \dot{x}(0)=y(0)=\dot{y}(0)=0$",fontsize=7) 
plt.legend()
plt.savefig(plotsDir+"Fig 4.png")

L = 1e-6
C = 1e-6
R = 100
q5Hs = sgnl.lti([1], [L*C, R*C, 1])
w, mag, phase = sgnl.bode(q5Hs)

plt.figure(5)
plt.semilogx(w, mag)
plt.xlabel(r"$\omega \ \to$")
plt.ylabel(r"$\|H(jw)\|\ (in\ dB)$")
plt.title("Magnitude plot of the given RLC network")
plt.savefig(plotsDir+"Fig 5(a).png")

plt.figure(6)
plt.xlabel(r"$\omega \ \to$")
plt.ylabel(r"$\angle H(jw)\ (in\ ^o)$")
plt.title("Phase plot of the given RLC network")
plt.semilogx(w, phase)
plt.savefig(plotsDir+"Fig 5(b).png")

t = np.linspace(0, 0.1, 1e6)
vi = np.cos(1e3*t) - np.cos(1e6*t)
taxis, vout, rest = sgnl.lsim(q5Hs, vi, t)

plt.figure(7)
plt.plot(taxis, vout)
plt.xlabel(r"$t\ \to$")
plt.ylabel(r"$v_o(t)\ \to$")
plt.title(r"$v_o(t)$" " given $v_i(t)=cos(10^3t)-cos(10^6t)$")
plt.savefig(plotsDir+"Fig 6(a).png")

plt.figure(8)
plt.plot(taxis, vout)
plt.xlim(0, 3e-5)
plt.ylim(0, 0.35)
plt.xlabel(r"$t\ \to$")
plt.ylabel(r"$v_o(t)\ \to$")
plt.title(r"$v_o(t)$ for $t<30\ us$")
plt.savefig(plotsDir+"Fig 6(b).png")
