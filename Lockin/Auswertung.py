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

sys.path.append("..\globales\python")
import latextables as lxtabs

usqrt = unc.wrap(np.sqrt)

U0 = 10e-03  # [V]

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


uU1_0= unp.uarray(np.zeros(len(uU1)), np.zeros(len(uU1)))
for i in range(len(uU1)):
   if not (np.sin(p1[i]) == 0):
       uU1_0[i] =  (const.pi/2) * (uU1[i]/np.sin(p1[i]))
   else:
       uU1_0[i] = ufloat(0,0)
       
uU1_0_new = np.delete(uU1_0, (0, 6))      
 
uU1_0_avr = np.mean(uU1_0_new)

uU1_0_std = np.std(noms(uU1_0_new))/sqrt(np.alen(uU1_0_new))
uU1_0_avr = ufloat(noms(uU1_0_avr),uU1_0_std)

print(uU1_0_avr)
      
       
       
# Plot der Messwerte
def S(x, a):
    return np.sin(x) * a

popt1, pcov1 = curve_fit(S, p1, noms(uU1), sigma=stds(uU1))
error1 = np.sqrt(np.diag(pcov1))

uU1_out = ufloat(popt1[0], error1[0])

plt.clf()
plt.grid()

x = np.linspace(-2*const.pi, 6*const.pi, num=1000)
plt.xlabel(r"Phase $\phi$")
plt.ylabel(r"Spannung $U_{out}\,[\mathrm{V}]$")
plt.ylim(-2e-02, 2e-02)
plt.xlim(0.5, 2*const.pi)
plt.xticks((-pi/3, 0, pi/3, 2*pi/3, pi, 4*pi/3, 5*pi/3, 2*pi),
          (r"$-\frac{\pi}{3}$", r"$0$",
           r"$\frac{\pi}{3}$", r"$\frac{2\pi}{3}$",
           r"$\pi$", r"$\frac{4\pi}{3}$", r"$\frac{5\pi}{3}$",
           r"$2\pi$"))
plt.tick_params(axis='x', which='both', labelsize=16)
plt.errorbar(p1, noms(uU1), yerr=stds(uU1), fmt="rx", label="Messwerte")
plt.plot(x, S(x, popt1), color="gray", label="Ausgleichskurve")
plt.legend(loc="best")
plt.savefig("Grafiken/OhneNoise.pdf")

# Messwerte mit Rauschen Phase, Spannung ,Gain

p2, U2, g2 = np.loadtxt("Messdaten/MitRauschen.txt", unpack=True)


# Fehlerbehaftete Spannung
uU2 = unp.uarray(U2, len(U2)*[U_err])


# Einheiten und Skalierung
p2 = np.deg2rad(p2)  # [rad]
uU2 /= g2  # [V]

uU2_0= unp.uarray(np.zeros(len(uU2)), np.zeros(len(uU2)))
for i in range(len(uU2)):
   if not (np.sin(p2[i]) == 0):
       uU2_0[i] = -1* (const.pi/2) * (uU2[i]/np.sin(p2[i]))
   else:
       uU2_0[i] = ufloat(0,0)
       
uU2_0_new = np.delete(uU2_0, (0, 6))      
 
uU2_0_avr = np.mean(uU2_0_new)

uU2_0_std = np.std(noms(uU2_0_new))/sqrt(np.alen(uU2_0_new))
uU2_0_avr = ufloat(noms(uU2_0_avr),uU2_0_std)

print(uU2_0_avr)

# Plot der Messwerte

popt2, pcov2 = curve_fit(S, p2, noms(uU2), sigma=stds(uU2))
error2 = np.sqrt(np.diag(pcov2))

uU2_out = ufloat(popt2[0], error2[0])

plt.clf()
plt.grid()

x = np.linspace(-2*const.pi, 6*const.pi, num=1000)
plt.xlabel(r"Phase $\phi$")
plt.ylabel(r"Spannung $U_{out}\,[\mathrm{V}]$")
plt.ylim(-1e-02, 1e-02)
plt.xlim(0.5, 2*const.pi)
plt.xticks((-pi/3, 0, pi/3, 2*pi/3, pi, 4*pi/3, 5*pi/3, 2*pi),
          (r"$-\frac{\pi}{3}$", r"$0$",
           r"$\frac{\pi}{3}$", r"$\frac{2\pi}{3}$",
           r"$\pi$", r"$\frac{4\pi}{3}$", r"$\frac{5\pi}{3}$",
           r"$2\pi$"))
plt.tick_params(axis='x', which='both', labelsize=16)
plt.errorbar(p2, noms(uU2), yerr=stds(uU2), fmt="rx", label="Messwerte")
plt.plot(x, S(x, popt2), color="gray", label="Ausgleichskurve")
plt.legend(loc="best")
plt.savefig("Grafiken/MitNoise.pdf")

# Messerter für die Abstandsmessung Abstand, Spannung, Gain
R3, U3, g3 = np.loadtxt("Messdaten/Abstand.txt", unpack=True)
U3 = np.abs(U3)
R3 += R_off

# Fehlerbehaftete Spannung
uU3 = unp.uarray(U3, len(U3)*[U_err])

# Fehlerbehafteter Abstand
uR3 = unp.uarray(R3, R_err)

uR3 *= 1e-02

# Einheiten und Skalierung
uU3 /= g3  # [V]



# Plot der Messwerte
def F(x, a):
    return a/(x**2)

popt3, pcov3 = curve_fit(F, noms(R3), noms(uU3), sigma=stds(uU3))
error3 = np.sqrt(np.diag(pcov3))

uU3_out = ufloat(popt3[0], error3[0])

plt.clf()
plt.grid()

r = np.linspace(1e-02,100 ,num=1000)

plt.ylim(0, 2e-02)
plt.xlim(0, 50)
plt.errorbar(noms(R3), noms(uU3), yerr=stds(uU3), fmt="rx", label="Messwerte")
plt.plot(r, F(r, *popt3), color="gray", label="Ausgleichskurve")
plt.legend(loc="best")
plt.savefig("Grafiken/Abstand.pdf")
print(popt3, error3)


## Print Funktionen 
PRINT = True
if PRINT:
    print("\nOutput Amplitude ohne Rauschen:\n", uU1_out)
    print("\nOutput Amplitude mit Rauschen:\n", uU2_out)


TABS = True
#if TABS:
#    f = open("Daten/Tabelle_ohneNoise.tex", "w")
#
#    f.write(lxtabs.toTable([np.rad2deg(p1), uU1, uU1_theo],
#        col_titles=["Phase", "Spannung", "Spannung"],
#        col_syms=[r"\Delta\phi", "U_{out}", "U_{out,theo}"],
#        col_units=[r"\degree", r"\volt", r"\volt"],
#        fmt=["c", "c", "c"],
#        cap="Messwerte der Messung ohne Noise-Generator",
#        label="ohneNoise"))
#
#    f.close()
#
#    f1 = open("Daten/Tabelle_ohneNoise.tex", "w")
#
#    f1.write(lxtabs.toTable([np.rad2deg(p1), uU1, uU1_0],
#        col_titles=["Phase", "Spannung", "Spannung"],
#        col_syms=[r"\Delta\phi", "U_{out}", "U_{0}"],
#        col_units=[r"\degree", r"\volt", r"\volt"],
#        fmt=["c", "c", "c"],
#        cap="Messwerte der Messung ohne Noise-Generator",
#        label="ohneNoise"))
#
#    f1.close()
#    f1 = open("Daten/Tabelle_mitNoise.tex", "w")
#
#    f1.write(lxtabs.toTable([np.rad2deg(p2), uU2, uU2_0],
#        col_titles=["Phase", "Spannung", "Spannung"],
#        col_syms=[r"\Delta\phi", "U_{out}", "U_{0}"],
#        col_units=[r"\degree", r"\volt", r"\volt"],
#        fmt=["c", "c", "c"],
#        cap="Messwerte der Messung mit Noise-Generator",
#        label="mitNoise"))
#
#    f1.close()
#    f1 = open("Daten/Tabelle_Abstand.tex", "w")
#
#    f1.write(lxtabs.toTable([uR3, uU3],
#        col_titles=["Abstand", "Spannung"],
#        col_syms=["r", "U_{out}"],
#        col_units=[r"\meter", r"\volt"],
#        fmt=["c", "c"],
#        cap="Messwerte der Intensität im Abstand r",
#        label="mitNoise"))
#
#    f1.close()