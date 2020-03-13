'''
------------------------------------
Assignment 7 - EE2703 (Jan-May 2020)
Done by Akilesh Kannan (EE18B122)
Created on 12/03/20
Last Modified on 13/03/20
------------------------------------
'''

import numpy as np
import scipy.signal as sgnl
import matplotlib.pyplot as plt
import sympy as sym

PI = np.pi
s = sym.symbols('s')
plotsDir='plots/'

plt.rcParams['figure.figsize'] = 18, 6
plt.rcParams['font.family'] = "sans"

def lowPass(R1, R2, C1, C2, G, Vi):
    s = sym.symbols('s')
    A = sym.Matrix([[0, 0, 1, -1/G],
                    [-1/(1+s*R2*C2), 1, 0, 0],
                    [0, -G, G, 1],
                    [-1/R1-1/R2-s*C1, 1/R2, 0, s*C1]])
    b = sym.Matrix([0,
                    0,
                    0,
                    -Vi/R1])
    V = A.inv() * b
    return V[3]

def sympyToLTI(symboFunc):
    numer, denom = symboFunc.as_numer_denom()
    numer = sym.Poly(numer, s)
    denom = sym.Poly(denom, s)
    numeratorCoeffs = numer.all_coeffs()
    denominatorCoeffs = denom.all_coeffs()
    for i in range(len(numeratorCoeffs)):
        x = float(numeratorCoeffs[i])
        numeratorCoeffs[i] = x
    for j in range(len(denominatorCoeffs)):
        x = float(denominatorCoeffs[j])
        denominatorCoeffs[j] = x
    return numeratorCoeffs, denominatorCoeffs

# Question 1
Vo = lowPass(10000, 10000, 1e-9, 1e-9, 1.586, 1)
voNumerator, voDenominator = sympyToLTI(Vo)
ckt1 = sgnl.lti(voNumerator, voDenominator)

## Bode Plot of Transfer Function
w, mag, phase = sgnl.bode(ckt1, w=np.linspace(1, 1e6, 1e6))
fig1 = plt.figure(1)
fig1.suptitle(r'Bode Plot of Transfer function of lowpass filter')
plt.subplot(211)
plt.semilogx(w, mag)
plt.ylabel(r'$20log(\|H(j\omega)\|)$')
plt.subplot(212)
plt.semilogx(w, phase)
plt.xlabel(r'$\omega \ \to$')
plt.ylabel(r'$\angle H(j\omega)$')
plt.savefig(plotsDir+'Fig 1.png')

## Calculate step response
t, voStep = sgnl.step(ckt1, None, np.linspace(0, 0.1, 1000000))
plt.figure(2)
plt.plot(t, voStep)
plt.title(r'Step Response of Lowpass filter')
plt.xlabel(r'$t\ \to$')
plt.ylabel(r'$V_o(t)\ \to$')
plt.xlim(0, 1e-3)
plt.grid(True)
plt.savefig(plotsDir+'Fig 2.png')

# Question 2
vi = np.heaviside(t,1)*(np.sin(2e3*PI*t)+np.cos(2e6*PI*t))

plt.figure(3)
plt.plot(t, vi)
plt.title(r'$V_i(t)=(sin(2x10^3\pi t)+cos(2x10^6\pi t))u(t)$ to Lowpass filter')
plt.xlabel(r'$t\ \to$')
plt.ylabel(r'$V_i(t)\ \to$')
plt.xlim(0, 1e-3)
plt.grid(True)
plt.savefig(plotsDir+'Fig 3.png')

time, vOut, rest = sgnl.lsim(ckt1, vi, np.linspace(0, 1e-1, 1e6))
plt.figure(4)
plt.plot(time, vOut)
plt.title(r'$V_o(t)$ for $V_i(t)=(sin(2x10^3\pi t)+cos(2x10^6\pi t))u(t)$ for Lowpass filter')
plt.xlabel(r'$t\ \to$')
plt.ylabel(r'$V_o(t)\ \to$')
plt.xlim(0, 1e-3)
plt.grid(True)
plt.savefig(plotsDir+'Fig 4.png')

# Question 3
def highPass(R1, R3, C1, C2, G, Vi):
    s = sym.symbols('s')
    A = sym.Matrix([[0, -1, 0, 1/G],
                    [s*C2*R3/(s*C2*R3+1), 0, -1, 0],
                    [0, G, -G, 1],
                    [-s*C2-1/R1-s*C1, 0, s*C2, 1/R1]])

    b = sym.Matrix([0,
                    0,
                    0,
                    -Vi*s*C1])
    return (A.inv()*b)[3]

Vo = highPass(10000, 10000, 1e-9, 1e-9, 1.586, 1)
voNumerator, voDenominator = sympyToLTI(Vo)
ckt2 = sgnl.lti(voNumerator, voDenominator)

## Bode Plot of Transfer Function
w, mag, phase = sgnl.bode(ckt2, w=np.linspace(1, 1e6, 1e6))
fig5 = plt.figure(5)
fig5.suptitle(r'Bode Plot of Transfer function of highpass filter')
plt.subplot(211)
plt.semilogx(w, mag)
plt.ylabel(r'$20log(\|H(j\omega)\|)$')
plt.subplot(212)
plt.semilogx(w, phase)
plt.xlabel(r'$\omega \ \to$')
plt.ylabel(r'$\angle H(j\omega)$')
plt.savefig(plotsDir+'Fig 5.png')

## Calculate step response
t, voStep = sgnl.step(ckt2, None, np.linspace(0, 0.1, 1000000))
plt.figure(6)
plt.plot(t, voStep)
plt.title(r'Step Response of Highpass filter')
plt.xlabel(r'$t\ \to$')
plt.ylabel(r'$V_o(t)\ \to$')
plt.xlim(0, 1e-3)
plt.grid(True)
plt.savefig(plotsDir+'Fig 6.png')

vi = np.heaviside(t,1)*(np.sin(2e3*PI*t)+np.cos(2e6*PI*t))

plt.figure(7)
plt.plot(t, vi)
plt.title(r'$V_i(t)=(sin(2x10^3\pi t)+cos(2x10^6\pi t))u(t)$ to Highpass filter')
plt.xlabel(r'$t\ \to$')
plt.ylabel(r'$V_i(t)\ \to$')
plt.xlim(0, 1e-3)
plt.grid(True)
plt.savefig(plotsDir+'Fig 7.png')

time, vOut, rest = sgnl.lsim(ckt2, vi, np.linspace(0, 1e-1, 1e6))
plt.figure(8)
plt.plot(time, vOut)
plt.title(r'$V_o(t)$ for $V_i(t)=(sin(2x10^3\pi t)+cos(2x10^6\pi t))u(t)$ for Highpass filter')
plt.xlabel(r'$t\ \to$')
plt.ylabel(r'$V_o(t)\ \to$')
plt.xlim(0, 1e-3)
plt.grid(True)
plt.savefig(plotsDir+'Fig 8.png')

plt.show()
