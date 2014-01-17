# -*- coding: utf-8 -*-
"""
Created on Fri Nov 08 23:56:02 2013

@author: Josh

"""

from __future__ import (print_function,
                        division,
                        unicode_literals,
                        absolute_import)
import math
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

sys.path.append("..\_globales\python")
import latextables as lxtabs

def f(x, a, b):
    return a*np.exp(b * x)


# Laden der Messdaten der Aufladung
t_auf, U_auf = np.loadtxt("Messdaten/Aufladen.txt", unpack=True)


# Laden der Messdaten der Entladung
t_ent, U_ent = np.loadtxt("Messdaten/Entladen.txt", unpack=True)

# Regression der Messwerte
popt1, pcov1 = curve_fit(f, t_ent, U_ent)
m1 = popt1[1]
b1 = np.log(popt1[0])

# Berechnung der Zeitkonstante
RC1 = - 1/m1

# Plot der Messwerte und Reressionskurve
t = np.linspace(-0.0001, 0.0009, num=1000)
plt.clf()
plt.grid()
plt.xlim(-0.0001, 0.0009)
plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter
                                    (lambda x, _: float(x * 1e3)))
plt.ylim(1e00,  1e03)
plt.xlabel("Zeit $t\,[ms]$")
plt.ylabel("Kondensatorspannung $U_{C}\,[V]$")
plt.yscale("log")
plt.plot(t_ent, U_ent, "rx", label="Messdaten")
plt.plot(t, f(t, *popt1), color="gray", label="Regressionsgerade")
plt.legend(loc="best")
plt.tight_layout()

# Laden der Frequenzabhängigen Amplituden
f1, Amps1 = np.loadtxt("Messdaten/Frequenzen_Amplituden_1.txt", unpack=True)
f2, Amps2 = np.loadtxt("Messdaten/Frequenzen_Amplituden_2.txt", unpack=True)

# Laden der Fehler
Amps1_err, Amps2_err = np.loadtxt("Messdaten/Amplituden_Fehler.txt",
                                  unpack=True)


## Print Funktionen