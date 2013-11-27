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



#
#t, a, b, c = var("T A B C")
#dp = 3 * a * t**2 + 2*b*t + c
#print(r"\sigma{\od{p}{T}}=", error(dp))

dp, vd, T = var("\diff{p} V_{D} T")
L = dp * vd * T 
print(r"\sigma_{L}=", error(L)  )




