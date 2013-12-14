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

#sys.path.append("..\globales\python")
#import latextables as lxtabs

# Messfehler Spannung, Abstand
U_err, R_err = np.loadtxt("Messdaten/Fehler.txt")

# Offset Abstand
R_off = np.loadtxt("Messdaten/AbstandOffset.txt")

# Messwerte ohne Rauschen: Phase, Spannung, Gain

p1, U1, g1 = np.loadtxt("Messdaten/OhneRauschen.txt", unpack=True)


# Fehlerbehaftete Spannung
uU1 = unp.uarray(U1, len(U1)*[U_err])

# Einheiten und Skalierung
p1 = np.deg2rad(p1)  # [rad]
uU1 /= g1  # [V]


# Plot der Messwerte
def S(x, a):
    return np.sin(x) * a

popt1, pcov1 = curve_fit(S, p1, noms(uU1), sigma=stds(uU1))
error1 = np.sqrt(np.diag(pcov1))

uU1_out = ufloat(popt1[0], error1[0])

plt.clf()
plt.grid()

x = np.linspace(-2*const.pi, 6*const.pi, num=1000)
plt.ylim(-2e-02, 2e-02)
plt.xlim(0.5, 2*const.pi)
plt.xticks((-pi/3, 0, pi/3, 2*pi/3, pi, 4*pi/3, 5*pi/3, 2*pi),
          (r"$-\frac{\pi}{3}$", r"$0$",
           r"$\frac{\pi}{3}$", r"$\frac{2\pi}{3}$",
           r"$\pi$", r"$\frac{4\pi}{3}$", r"$\frac{5\pi}{3}$",
           r"$2\pi$"))
plt.errorbar(p1, noms(uU1), yerr=stds(uU1), fmt="rx")
plt.plot(x, S(x, popt1), color="gray")


# Messwerte mit Rauschen Phase, Spannung ,Gain

p2, U2, g2 = np.loadtxt("Messdaten/MitRauschen.txt", unpack=True)


# Fehlerbehaftete Spannung
uU2 = unp.uarray(U2, len(U2)*[U_err])


# Einheiten und Skalierung
p2 = np.deg2rad(p2)  # [rad]
uU2 /= g2  # [V]


# Plot der Messwerte

popt2, pcov2 = curve_fit(S, p2, noms(uU2), sigma=stds(uU2))
error2 = np.sqrt(np.diag(pcov2))

uU2_out = ufloat(popt2[0], error2[0])

plt.clf()
plt.grid()

x = np.linspace(-2*const.pi, 6*const.pi, num=1000)
plt.ylim(-1e-02, 1e-02)
plt.xlim(0.5, 2*const.pi)
plt.xticks((-pi/3, 0, pi/3, 2*pi/3, pi, 4*pi/3, 5*pi/3, 2*pi),
          (r"$-\frac{\pi}{3}$", r"$0$",
           r"$\frac{\pi}{3}$", r"$\frac{2\pi}{3}$",
           r"$\pi$", r"$\frac{4\pi}{3}$", r"$\frac{5\pi}{3}$",
           r"$2\pi$"))
plt.errorbar(p2, noms(uU2), yerr=stds(uU2), fmt="rx")
plt.plot(x, S(x, popt2), color="gray")


# Messerter für die Abstandsmessung Abstand, Spannung, Gain
R3, U3, g3 = np.loadtxt("Messdaten/Abstand.txt", unpack=True)
U3 = np.abs(U3)
R3 += R_off

# Fehlerbehaftete Spannung
uU3 = unp.uarray(U3, len(U3)*[U_err])

# Fehlerbehafteter Abstand
uR3 = unp.uarray(R3, R_err)


# Einheiten und Skalierung
uU3 /= g3  # [V]



# Plot der Messwerte
def F(x, a, n):
    return a/(x**n)

popt3, pcov3 = curve_fit(F, noms(R3), noms(uU3), sigma=stds(uU3))
error3 = np.sqrt(np.diag(pcov3))

uU3_out = ufloat(popt3[0], error3[0])

plt.clf()
plt.grid()
r = np.linspace(1e-02,100 ,num=1000)
plt.ylim(0, 2e-02)
plt.xlim(0, 40)
plt.errorbar(noms(R3), noms(uU3), yerr=stds(uU3), fmt="rx")
plt.plot(r, F(r, *popt3), color="gray")

print(popt3)


## Print Funktionen 
PRINT = True
if PRINT:
    print("\nOutput Amplitude ohne Rauschen:\n", uU1_out)
    print("\nOutput Amplitude mit Rauschen:\n", uU2_out)
