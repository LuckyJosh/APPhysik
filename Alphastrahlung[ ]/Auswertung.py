# -*- coding: utf-8 -*-
"""
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
import uncertainties as unc
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)

from aputils.latextables.tables import Table

pulse = np.loadtxt("Messdaten/MessreiheIII.txt")
ranges = np.arange(0, 13087, 500)
Lists = []

for i in range(len(ranges)):
    if i == len(ranges) -1:
        break
    List = []
    for p in pulse:
        if ranges[i] < p < ranges[i + 1]:
            List.append(p)
    Lists.append(List)

for l in Lists:
    print(l, len(l))

Sum = 0
for j in range(len(ranges)):
    if not j == len(ranges) - 1:
        Sum += len(Lists[j])
        plt.bar(ranges[j], len(Lists[j]))
plt.show()
plt.e
print(Sum)
## Print Funktionen
T = Table()
