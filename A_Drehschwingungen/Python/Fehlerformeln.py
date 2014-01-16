# -*- coding: utf-8 -*-
"""
Created on Tue Nov 05 23:17:34 2013

@author: Josh
"""


from __future__ import (print_function,
                        division,
                        unicode_literals,
                        absolute_import)
import math as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit
from sympy import *  
from uncertainties import ufloat
import uncertainties.unumpy as unp


def error(f, err_vars=None):
    from sympy import Symbol, latex
    s = 0
    latex_names = dict()

    if err_vars is None:
        err_vars = f.free_symbols

    for v in err_vars:
        err = Symbol('latex_std_' + v.name)
        s += f.diff(v)**2 * err**2
        latex_names[err] = '\\sigma_{' + latex(v) + '}'
    return latex(sqrt(s), symbol_names=latex_names)


m, r = var("m_K r_K")
I_K = 0.4 * m * r**2
print("I_K", error(I_K))

L, T, R, I, I_h = var("L T R I_K I_H")
G = ((8 * const.pi * L) / (T**2 * R**4)) * (I + I_h)
print("G", error(G, (L, T, R, I)))

E, G, mu = var("E G \mu")
P = (E / (2 * G)) - 1
print("P", error(P))
Q = E/(3 * (1 - 2 * mu))
print("Q", error(Q))


b = var("B")
m = (4 * const.pi**2 * (I + I_h)/(b * T**2) -
     ((const.pi * G * (R)**4)/(2 * L * B)))
print("m:", error(m, (I, T, B, G, R, L)))


N, I, R = var("N I R")
B = const.mu_0 * (8/ma.sqrt(125)) * (N * I) / R
print("B:", error(B, (N, I, R)))
print((const.mu_0 * (8/ma.sqrt(125)))**2)







