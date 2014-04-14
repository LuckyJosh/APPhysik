# -*- coding: utf-8 -*-
"""
Created on Thu Apr 03 02:24:01 2014

@author: Josh
"""

from __future__ import (print_function,
                        division,
                        unicode_literals,
                        absolute_import)

import math
import numpy as np
import uncertainties as unc
import uncertainties.unumpy as unp
from sympy import *


class Quantity:
    def __init__(self, list_, err=None, factor=1):
        self.avr = np.mean(list_)*factor
        self.std = np.std(list_)*factor
        self.len = len(list_)
        self.std_err = self.std / math.sqrt(self.len)
        self.avr_err = unc.ufloat(self.avr, self.std_err)*factor

        if not err is None:
            Umean = unc.wrap(np.mean)
            self.list_err = unp.uarray(list_, [err]*self.len)*factor
            self.avr_err_gauss = Umean(self.list_err)*factor


class error:
    def __init__(f, symbol="", err_vars=None):
        self.var_equation = 0
        self.latex_names = dict()
        self.err_vars = err_vars
        self.function = f

        if self.err_vars is None:
            self.err_vars = function.free_symbols

        for v in self.err_vars:
            err = Symbol('latex_std_' + v.name)
            self.error_equation += self.function.diff(v)**2 * err**2
            self.latex_names[err] = '\\sigma_{' + latex(v) + '}'

        self.std = ('\\sigma_{' + symbol + '}=' +
                    latex(sqrt(s), symbol_names=latex_names))

    def show():
        pass



def error(f, symbol="", err_vars=None):
    s = 0
    latex_names = dict()

    if err_vars is None:
        err_vars = f.free_symbols

    for v in err_vars:
        err = Symbol('latex_std_' + v.name)
        s += f.diff(v)**2 * err**2
        latex_names[err] = '\\sigma_{' + latex(v) + '}'
    return ('\\sigma_{' + symbol + '}=' +
            latex(sqrt(s), symbol_names=latex_names))

if __name__ == "__main__":
    l1, l2, l3 = np.loadtxt("../testsdir/test.txt", unpack=True)
    L1 = Quantity(l1)
    L2 = Quantity(l2, err=0.5)
    L3 = Quantity(l3, err=1, factor=1e03)
    x, y, z = var("x y z")
    F = x**2 + y**2 + z**3
    print(error(F,symbol="F"))




