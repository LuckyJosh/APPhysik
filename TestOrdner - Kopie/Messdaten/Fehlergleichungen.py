# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 17:20:46 2014

@author: JoshLaptop
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


l, t = var("l t")
V = l / t
print(error(V))

lam, nu = var(r"\lambda \nu")
C = lam * nu
print(error(C))

nu0 = var(r"\nu0")
dnu = nu - nu0
print(error(dnu))


