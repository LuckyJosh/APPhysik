# -*- coding: utf-8 -*-
"""
Created on Wed Jun 04 16:54:36 2014

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
import uncertainties as unc
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)


def distToLogFmt(dists, scale):
    tens = np.trunc(dists/10)
    ones = dists % 10
    logfmt = np.zeros(len(dists))
    for i in range(len(dists)):
        logfmt[i] = (tens[i] + np.log10((tens[i]*10 + 1)/(tens[i]*10)) * scale[tens[i]-1]
              + ((scale[tens[i]-1] *
              (1 - np.log10((tens[i]*10 + 1)/(tens[i]*10)))) * np.log10(ones[i])))
    return logfmt

def LogFmtToDist(logfmt, scale):
    tens = np.trunc(logfmt)
    ones = logfmt - tens
    dists= np.zeros(len(logfmt))
    result = np.zeros(len(logfmt))
    for i in range(len(dists)):
        if tens[i] >= 10:
            adder = 1
            scaler = 0
            if i == len(logfmt) - 1:
                ones[i] *=10
        else:
            adder = 0.1
            scaler = tens[i] - 1
        dists[i] = 10**((ones[i] - scale[scaler] * np.log10((tens[i] + adder)/tens[i]))/
                       (scale[scaler] - scale[scaler] * np.log10((tens[i] + adder)/tens[i])))


        if tens[i] >= 10:
            result[i] = tens[i] + dists[i]
        else:
            result[i] = tens[i] + dists[i]/10

    return result




# Abstände der X-Achse
d_X = np.loadtxt("X_Abstand.txt")

# Abstände der Y-Achse
d_Y = np.loadtxt("Y_Abstand.txt")

# Abstandsmesswerte
L = np.loadtxt("Abstaende.txt")

# Laden der Intensität in logfmt
I = np.loadtxt("Intensitaeten.txt")


L_logfmt = np.round(distToLogFmt(L, d_X), 2)

dists = LogFmtToDist(I, d_Y)

plt.loglog(L, dists, "xr", basex=10, basey=10)
plt.grid(which="both")
plt.ylim(1, 30)
plt.xlim(10, 200)
plt.show()

#print(d_X)
#print(d_Y)
#print(L)
