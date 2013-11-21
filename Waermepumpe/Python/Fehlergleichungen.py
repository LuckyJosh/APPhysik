# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 17:30:11 2013

@author: Josh
"""

from __future__ import (print_function,
                        division,
                        unicode_literals,
                        absolute_import)
import math as m
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit
from sympy import *
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms,
                                  std_devs as stds)

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


m, c, M, C, dT1, P = var("m_1 c_w m_k c_k dT1 P")

Nu_real = (m*c + M*C)/P * dT1

print(error(Nu_real,(P, dT1)))
T1, T2, L = var("T_1 T_2 L")

Nu_id = T1/(T1-T2)
print("\n ideal:\n", error(Nu_id))


dm = (m*c + M*C)/L * dT1

print("\n Massendurchsatz:\n", error(dm, (L, dT1)))

A, B, T = var("A B t")
dT = 2*A*T + B
print("\n Temperaturdifferentialquotient:\n", error(dT, (A, B)))








