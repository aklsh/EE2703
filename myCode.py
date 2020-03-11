'''
-------------------------------------
 Assignment 7 - EE2703 (Jan-May 2020)
 Done by Akilesh Kannan (EE18B122)
 Created on 11/03/20
 Last Modified on 11/03/20
-------------------------------------
'''

import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import scipy.signal as sgnl


def ckt1(R1, R2, C1, C2, G, Vi):
    s = sym.Symbol("s")
    A = sym.Matrix([[0, 0, 1, -1/G],
                    [-1/(1+s*R2*C2), 1, 0, 0],
                    [0, -G, G, 1],
                    [-1/R1-1/R2-s*C1, 1/R2, 0, s*C1]])
    b = sym.Matrix([0,
                    0,
                    0,
                    Vi/R1])
    x = (A**-1)*b
    return x[3]


sym.pprint(ckt1(10000, 10000, 1e-9, 1e-9, 1.586, 1))
