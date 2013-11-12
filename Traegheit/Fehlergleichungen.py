# -*- coding: utf-8 -*-
"""
Created on Thu Nov 07 15:55:14 2013

@author: JoshLaptop
"""

from __future__ import (print_function,
                        division,
                        unicode_literals,
                        absolute_import)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit
from scipy.stats import linregress
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



T, D, b = var("T D b")

I_D = b * D / (4 * const.pi**2)
print("EigenTrägheitsmoment", error(I_D))

I = T**2 * D / (4 * const.pi**2)
print("allgTrägheitsmoment", error(I))
