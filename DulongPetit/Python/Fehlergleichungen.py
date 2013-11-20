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



U = var("U")
T = 25.157 * U - 0.19 * U**2
print("Fehler von T:\n", "\sigma_{T} =", error(T))

cw, mw, cgmg, Tm, Tw, mk, Tk = var(r"c_\ce{H2O} m_\ce{H2O} c_g*m_g \vartheta_m \vartheta_c m_k \vartheta_h")

c_k = (cw*mw + cgmg) * (Tm - Tw)/(mk* (Tk - Tm))
print("Fehler der Wärmekapazität des Kalorimeters:\n", "\sigma_{c_k} =",
      error(c_k, (mw, cgmg, Tm, Tw, mk, Tk)))






