# -*- coding: utf-8 -*-
"""
Created on Fri Nov 08 23:56:02 2013

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

## Mögliche Fit-Funktionen


def T_FitI(t, A, B, C):
    return A * t**2 + B * t + C


def dT_FitI(t, A, B):
    return 2 * A * t + B


def T_FitII(t, A, B, a):
    return A/(1 + B * t**a)


def T_FitIII(t, A, B, C, a):
    return ((A * t**a) / (1 + B * t**a)) + C


#%%
## Laden der Konstanten
# Laden des Zeitintervalls
DELTA_T = np.loadtxt("../Messdaten/Zeitintervall.txt")

# Laden der Füllmenge
V_H2O = np.loadtxt("../Messdaten/Fuellmenge.txt")

# Laden der Wärmekapazität
C_APP = np.loadtxt("../Messdaten/Waermekapazitaet.txt")

# Laden der Leistung
P_APP = np.loadtxt("../Messdaten/Leistung.txt")


## Laden der Messungen

# Laden der Temperaturen
T_1 = np.loadtxt("../Messdaten/Temperatur_T1.txt", unpack=True)
T_2 = np.loadtxt("../Messdaten/Temperatur_T2.txt", unpack=True)
T_err = np.loadtxt("../Messdaten/Temperatur_Fehler.txt")

 # Anzahl der Messwerte
T_dim = len(T_1)

 # SI-Einheiten
T_1 += 273.15  # K
T_2 += 273.15  # K

 # Fehlerbehaftete Größen
uT_1 = unp.uarray(T_1, T_dim * [T_err])
uT_2 = unp.uarray(T_2, T_dim * [T_err])


# Laden der Drücke
P_1 = np.loadtxt("../Messdaten/Druck_p1.txt", unpack=True)
P_2 = np.loadtxt("../Messdaten/Druck_p2.txt", unpack=True)
P_1_err = np.loadtxt("../Messdaten/Druck_p1_Fehler.txt")
P_2_err = np.loadtxt("../Messdaten/Druck_p2_Fehler.txt")

 # Anzahl der Messwerte
P_dim = len(P_1)

 # SI-Einheiten
P_1 *= 1e05  # Pa
P_2 *= 1e05  # Pa
P_1_err *= 1e05  # Pa
P_2_err *= 1e05  # Pa

 # Fehlerbehaftete Größen
uP_1 = unp.uarray(P_1, P_dim * [P_1_err])
uP_2 = unp.uarray(P_2, P_dim * [P_2_err])

#%%

## Plots der Temperaturverläufe

t = np.linspace(0, (T_dim - 1) * DELTA_T, num=T_dim)
x = np.linspace(0, (T_dim - 1) * DELTA_T, num=1000)

poptI_T1, pcovI_T1 = curve_fit(T_FitI, t, T_1)
poptII_T1, pcovII_T1 = curve_fit(T_FitII, t, T_1)
poptIII_T1, pcovIII_T1 = curve_fit(T_FitIII, t, T_1)

poptI_T2, pcovI_T2 = curve_fit(T_FitI, t, T_2)
poptII_T2, pcovII_T2 = curve_fit(T_FitII, t, T_2)     # desolater Fit
poptIII_T2, pcovIII_T2 = curve_fit(T_FitIII, t, T_2)  # desolater Fit

# 4 Werte der Ableitung T_1
dT_1 = np.zeros(4)
for i in range(4):
    dT_1[i] = dT_FitI(t[(i+1)*2], poptI_T1[0], poptI_T1[1])

# 4 Werte der Ableitung T_2
dT_2 = np.zeros(4)
for i in range(4):
    dT_2[i] = dT_FitI(t[(i+1)*2], poptI_T2[0], poptI_T2[1])

plt.clf()
plt.grid()
#plt.semilogy()
plt.xlabel(r"Zeit $t[\mathrm{s}]$")
plt.ylabel(r"Temperatur $T[\mathrm{K}]$")

plt.plot(t, T_1, "rx", label="Messwerte $T_{1}$")
plt.plot(x, T_FitI(x, *poptI_T1), "k-", label="Fit1 $T_{1}$")

#plt.plot(x, T_FitII(x, *poptII_T1), "k-", label="Fit2 $T_{1}$")
#plt.plot(x, T_FitIII(x, *poptIII_T1), "k-", label="Fit3 $T_{1}$")


plt.plot(t, T_2, "bx", label="Messwerte $T_{2}$")
plt.plot(x, T_FitI(x, *poptI_T2), "k-", label="Fit1 $T_{2}$")

#plt.plot(x, T_FitII(x, *poptII_T2), "k-", label="Fit2 $T_{2}$")
#plt.plot(x, T_FitIII(x, *poptIII_T2), "k-", label="Fit3 $T_{2}$")

plt.legend(loc="best")
plt.show()





## Ausgaben
print("Temperaturen:\n", "-Warm:\n", uT_1, "\n", "-Kalt:\n", uT_2)
print("Drücke:\n", "-Warm:\n", uP_1, "\n", "-Kalt:\n", uP_2)
print("Zeiten:\n", t)
print("Parameter:\n", "Fit1:\n", poptI_T1, "\n", "Fit2:\n", poptII_T1, "\n",
      "Fit3:\n", poptIII_T1)
print("Ableitungen von T_1:\n", dT_1, "\n", "Ableitungen von T_2:\n", dT_2)