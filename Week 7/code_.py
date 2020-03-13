'''
------------------------------------
Assignment 7 - EE2703 (Jan-May 2020)
Done by Akilesh Kannan (EE18B122)
Created on 12/03/20
Last Modified on 12/03/20
------------------------------------
'''

import numpy as np
import scipy.signal as sgnl
import matplotlib.pyplot as plt
import sympy as sym

PI = np.pi
s = sym.symbols('s')

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
    return (A, b, V[3])

def sympyToLTI(symboFunc):
    numer, denom = sym.simplify(symboFunc).as_numer_denom()
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
A, b, Vo = lowPass(10000, 10000, 1e-9, 1e-9, 1.586, 1)
voNumerator, voDenominator = sympyToLTI(Vo)
ckt1 = sgnl.lti(voNumerator, voDenominator)
# Calculate step response
t, voStep = sgnl.step(ckt1, None, np.linspace(0, 0.001, 10000))
plt.plot(t, voStep)
plt.xlabel(r'$t\ \to$')
plt.ylabel(r'$V_o(t)\ \to$')
plt.grid(True)
plt.show(block=False)


# Question 2
vi = np.heaviside(t,1)*(np.sin(2e3*PI*t)+np.cos(2e6*PI*t))

plt.figure(2)
plt.plot(t, vi)
plt.xlabel(r'$t\ \to$')
plt.ylabel(r'$V_i(t)\ \to$')
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

Vo = highPass(1e4, 1e4, 1e-9, 1e-9, 1.586, 1)
voNumerator, voDenominator = sympyToLTI(Vo)
ckt2 = sgnl.lti(voNumerator, voDenominator)
time, vOut, rest = sgnl.lsim(ckt2, vi, np.linspace(0, 1e-5, 1e4))
plt.figure(3)
plt.plot(time, vOut)
plt.xlabel(r'$t\ \to$')
plt.ylabel(r'$V_o(t)\ \to$')
plt.grid(True)
plt.show()
