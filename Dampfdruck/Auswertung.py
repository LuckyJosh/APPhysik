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
import uncertainties as unc
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)

#sys.path.append("D:\Eigene Dateien\Studium\Physik\AnfängerPraktikum\PraktikumRepo\Versuchsvorlage\Python")
#import LatexTables as tab


# Definierte Makros:
PRINT = True
SI = True

# Physikalische Konstanten
R = const.gas_constant

### Laden der Messdaten

## Mesfehler: T1, p11, p12(ab 82°C), T2, p2
T1_err, p1_err, T2_err, p2_err = np.loadtxt("Messdaten/Messfehler.txt")
if SI:
    p1_err *= 1e02  # Pa
    p2_err *= 1e05   # Pa

## Versuchsreihe 1: T und p
T1, p1 = np.loadtxt("Messdaten/Messung_Tp_1.txt", unpack=True)

# Umrechnung in SI-Einheiten:
if SI:
    T1 += 273  # K
    p1 *= 1e02  # Pa

# Erstellen der Fehlerbahafteten Größen
# der Fehler des Druckes ist ab dem 24. Wert, wegen der stärkeren
# Fluktuation um Faktor 10 größer
uT1 = unp.uarray(T1, len(T1)*[T1_err])
up11 = unp.uarray(p1[0:11], p1_err)
up12 = unp.uarray(p1[11:], p2_err)
up1 = np.append(up11, up12)

## Versuchsreihe 2: T und p
T2, p2 = np.loadtxt("Messdaten/Messung_Tp_2.txt", unpack=True)


# Umrechnung in SI-Einheiten:
if SI:
    T2 += 273.15 # K
    p2 *= 1e05  # Pa

# Erstellen der Fehler behafteten Größen
uT2 = unp.uarray(T2, len(T2) * [T2_err])
up2 = unp.uarray(p2, p2_err)


### Ploten der 1.Messung
## Fit Funktion p
# TODO: Vllt nicht halblogarithmisch sondern den ln(p) ploten

def p(T, A, B):
    return A * np.exp(B/T)


popt1, pcov1 = curve_fit(p, noms(uT1), noms(up1), sigma=stds(up1))
error = np.sqrt(np.diag(pcov1))

# Erstellen der fehlerbehafteten Parameter
uParam_A = ufloat(popt1[0], error[0])
uParam_B = ufloat(popt1[1], error[1])

t = np.linspace(300, 500)

plt.clf()
plt.xlim(2.65e-03, 3.05e-03)
plt.ylim(1e04, 1e05)
plt.grid(which="both")
plt.yscale("log")
plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: float(x * 1e2)))
plt.gca().yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: float(x * 1e-05)))
plt.xlabel("reziproke Temperatur $T^{-1}\,10^{-2}[\mathrm{K^{-1}}]$")
plt.ylabel("Druck $p\,[\mathrm{bar}]$")
plt.errorbar(noms(1/uT1), noms(up1), xerr=stds(1/uT1),
             yerr=stds(up1), fmt="rx", label="Messwerte")
plt.plot(1/t, p(noms(t), *popt1), color="grey",
         label="Regessionsgerade")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("Grafiken/Messreihe_1.pdf")

## Bestimmung der Verdampungswärme: B = -L/R
uL = -1 * uParam_B * R
uL = ufloat(noms(uL), stds(uL))

## Berechnung der äußeren Verdampfungswärme L_a = RT
T = 373  # K
L_a = R * T


## Berechnung der inneren Verdampfungswäreme L_i = L - L_a
uL_i = uL - L_a
uL_i = ufloat(noms(uL_i), stds(uL_i))

# Pro Molekül in eV
uL_i_N = uL_i / const.Avogadro 
uL_i_eV = uL_i_N / const.electron_volt

### Plot der 2. Messreihe

# Fit-Ploynom 3.Grades


def Pg5(T, A, B, C, D, E, F):
    return A * T**5 + B * T**4 + C * T**3 + D * T**2 + E*T + F


def P(T, C, D, E, F):
    return C * T**3 + D * T**2 + E * T + F


popt2, pcov2 = curve_fit(P, noms(uT2), noms(up2), sigma=stds(up2))
error2 = np.sqrt(np.diag(pcov2))

t = np.linspace(300, 500, num=1000)

plt.clf()
plt.grid()
plt.ylim(0, 16e05)
plt.gca().yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: float(x * 1e-05)))
plt.xlabel("Temperatur $T\,[\mathrm{K}]$")
plt.xlabel("Druck $p\,[\mathrm{bar}]$")

plt.errorbar(noms(uT2), noms(up2), xerr=stds(uT2), yerr=stds(up2), fmt="rx", 
             label="Messwerte")
plt.plot(t, P(t, *popt2), color="grey", label="Regressionspolynom 3.Grades")

plt.legend(loc="best")
plt.tight_layout()
plt.savefig("Grafiken/Messreihe_2.pdf")


## Erstellen der Fehler behafteten Parameter
uParam_C = ufloat(popt2[0], error2[0])
uParam_D = ufloat(popt2[1], error2[1])
uParam_E = ufloat(popt2[2], error2[2])
uParam_F = ufloat(popt2[3], error2[3])

## Ableitung der Funktion P mit den Parametern


def dP(T):
    return 3 * noms(uParam_C) * T**2 + 2 * noms(uParam_D) * T + noms(uParam_E)
udP = unc.wrap(dP)


def L(dp, Vd, Vf, T):
    return dp * (Vd - Vf) * T
uLFunc = unc.wrap(L)

## Volumen des Dampf
Usqrt = unc.wrap(np.sqrt)
a = 0.9  # J m³/ mol²
V_D_1 = (R * T2 / (2 * p2)) + Usqrt((R * T2 / (2 * p2))**2 - (a / p2))
V_D_2 = (R * T2 / (2 * p2)) - Usqrt((R * T2 / (2 * p2))**2 - (a / p2))
# Bei unserem Versuchsaufbau ist das (Kleinere) Volumen V_D_2 realistischer
# da das größere Volumen mit bspw. 29 Literen nicht in den Kolben gepasst hätte


## Berechnung von L


def pg2(x, G, H, I):
    return G*x**2 + H*x + I

uL_2 = uLFunc(udP(uT2), V_D_2, 0, uT2)
uL_2 = unp.uarray(noms(uL_2), stds(uL_2))

popt3, pcov3 = curve_fit(pg2, noms(uT2), noms(uL_2), sigma=stds(uL_2))
error3 = np.sqrt(np.diag(pcov3))

T = np.linspace(200, 600)
# Plot der Verdampfungswärme
plt.clf()
plt.grid()
plt.xlim(300, 500)
plt.ylim(-500, 4000)
plt.xlabel(r"Temperatur $T\,[\mathrm{K}]$")
plt.ylabel(r"Verdampfungswärme $L\,[\mathrm{\frac{J}{mol}}]$")
plt.plot(noms(uT2), noms(uL_2), "rx", label="Werte")
plt.plot(T, pg2(T, *popt3), color="grey", label="Werte")
plt.plot()
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("Grafiken/Verlauf_LT.pdf")

# Erstellen der fehlerbehafteten Parameter
uParam_G = ufloat(popt3[0], error3[0])
uParam_H = ufloat(popt3[1], error3[1])
uParam_I = ufloat(popt3[2], error3[2])


### Print Funktionen


#print(toTable("Temperatur- und Druckmesswerte der ersten Versuchsreihe",
#              "DataI", [T1, p1]))
#print(np.where(T1 == 355))
#print(up1)

#PRINT = False

if PRINT:
    print("\nMessung 1:",
          "\nTemperatur 1:\n", uT1,
          "\n\nDruck 1:\n", up1)

    print("\nMessung 2:",
          "\nTemperatur 2:\n", uT2,
          "\n\nDruck 2:\n", up2)

    print("\nFit-Parameter 1:\n",
          "A =", uParam_A*1e-05, "\n",
          np.log(popt1[0]*1e-05), log(error[0]*1e-05),"\n",
          "B =", uParam_B*1e-05, "\n",
          popt1[1]*1e-05, error[1]*1e-05)

    print("\nVerdampfungswärmen:\n",
          "\ngemittelte Verdampfungswärme:\n", uL,
          "\näußere Verdampfungswärme:\n", L_a,
          "\ninnere Verdampfungswärme:\n", uL_i,
          "\ninnere pro Molekül und in eV:\n", uL_i_eV,
          "\nTemperaturabhängige:\n", uL_2)

    print("\nFit-Parameter:\n",
          "C =", uParam_C, "\n",
          "D =", uParam_D, "\n",
          "E =", uParam_E, "\n",
          "F =", uParam_F, "\n")

    print("\nDampfvolumen 1:\n", V_D_1, "unsinnvoll",
          "\nDampfvolumen 2:\n", V_D_2, "sinnvoll")

    print("\nFit-Parameter:\n",
          "G =", uParam_G, "\n",
          "H =", uParam_H, "\n",
          "I =", uParam_I, "\n")

#    print("\n" +
#          tab.toTable([uT1, up1*1e-02],
#          ["Temperatur", "Druck"],
#          ["T", "D"],
#          [r"\kelvin", r"\bar"],
#          cap="Messwerte der 1. Messung",
#          label="DataI"))
