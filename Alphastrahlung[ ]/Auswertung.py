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
from aputils.utils import Quantity


def func(x, a, b):
    return a*x + b

def effectiveLength(p, x_0, p_0 = 1013):
    return x_0 * (p / p_0)

def distanceToEnergy(R):
    return (R/(3.1))**(2/3)

### Aufgabenteil I
p_I, ch_I, pulse_I = np.loadtxt("Messdaten/MessreiheI.txt", unpack=True)
x_eff = effectiveLength(p_I, 20)
rate_I = pulse_I/120
half_max_I = max(rate_I)/2

popt, pcov = curve_fit(func, x_eff[0:-7], rate_I[0:-7])
errors = np.sqrt(np.diag(pcov))
a = ufloat(popt[0], errors[0])
b = ufloat(popt[1], errors[1])
print("FitparameterI:", a, b)




X = np.linspace(-10, 20, 1000)

f = [func(x, popt[0], popt[1]) for x in X]
lower = []
greater = []
for i in f:
    if i < half_max_I:
        lower.append(i)
    if i > half_max_I:
        greater.append(i)
intercept_y = (min(greater), max(lower))
intercept_x = X[np.where(f == min(greater))[0][0]]
print("Schnittpunkt x / Mittlere Reichweite:", intercept_x)
print("Schnittpunkt y:", half_max_I, intercept_y)
X_1 = np.linspace(-5, intercept_x)
distance_avr_I = intercept_x
energy_I = distanceToEnergy(distance_avr_I)
print("Energie der Alphateilchen:", energy_I)


plt.clf()
plt.xlim(-5, 20)
plt.ylim(0, 400)
plt.grid()
plt.xlabel("Effektive Länge $x\ [\mathrm{mm}]$", fontsize=14, family='serif')
plt.ylabel("Zerfallsrate $A\ [\mathrm{s^{-1}}]$", fontsize=14, family='serif')
plt.plot(X, len(X)*[half_max_I])
plt.plot(X, func(X, popt[0], popt[1]), label="Regressionsgerade", color="gray")
plt.plot(intercept_x, half_max_I, "r^")
plt.plot(x_eff, rate_I, "rx", label="Messwerte")
plt.tight_layout()
plt.legend(loc="best")
plt.show()

### Aufgabenteil II

p_II, ch_II, pulse_II = np.loadtxt("Messdaten/MessreiheII.txt", unpack=True)
x_eff_II = effectiveLength(p_I, 20)
rate_II = pulse_I/120
half_max_II = max(rate_II)/2

popt, pcov = curve_fit(func, x_eff_II[0:-5], rate_II[0:-5])
errors = np.sqrt(np.diag(pcov))
a = ufloat(popt[0], errors[0])
b = ufloat(popt[1], errors[1])
print("FitparameterII:", a, b)




X = np.linspace(-10, 20, 1000)

f = [func(x, popt[0], popt[1]) for x in X]
lower = []
greater = []
for i in f:
    if i < half_max_I:
        lower.append(i)
    if i > half_max_I:
        greater.append(i)
intercept_y = (min(greater), max(lower))
intercept_x = X[np.where(f == max(lower))[0][0]]
print("Schnittpunkt x / Mittlere Reichweite:", intercept_x)
print("Schnittpunkt y:", half_max_I, intercept_y)
X_1 = np.linspace(-5, intercept_x)
distance_avr_I = intercept_x
energy_I = distanceToEnergy(distance_avr_I)
print("Energie der Alphateilchen:", energy_I)

plt.clf()
plt.xlim(-5, 20)
plt.ylim(0, 400)
plt.grid()
plt.xlabel("Effektive Länge $x\ [\mathrm{mm}]$", fontsize=14, family='serif')
plt.ylabel("Zerfallsrate $A\ [\mathrm{s^{-1}}]$", fontsize=14, family='serif')
plt.plot(X, len(X)*[half_max_I])
plt.plot(X, func(X, popt[0], popt[1]), label="Regressionsgerade", color="gray")
plt.bar(intercept_x, half_max_I, width=0.1, alpha=0.3)
plt.plot(x_eff, rate_I, "rx", label="Messwerte")
plt.tight_layout()
plt.legend(loc="best")
plt.show()

### Aufgabenteil III

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2., 1.0*height, '%d'%int(height),
                 ha='center', va='bottom')

def poisson(x, mu):
    return mu**x * np.exp(-mu)/m.factorial(x)


pulse = np.loadtxt("Messdaten/MessreiheIII.txt")
pulse /= 10
Pulse_ges = Quantity(pulse)
print("Mittelwert, Abweichung:",Pulse_ges.avr, Pulse_ges.std_err)


ranges = np.arange(500, 1400, 40)
Lists = []
print(min(pulse), max(pulse), max(pulse) - min(pulse))
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

plt.clf()
Sum = 0
for j in range(len(ranges)):
    if not j == len(ranges) - 1 and not len(Lists[j]) == 0:
        Sum += len(Lists[j])
        rect = plt.bar(ranges[j], len(Lists[j]),
                       width=20, color="gray", alpha=0.5)#, label="Messwerte")
#        autolabel(rect)
else:
    plt.bar(1400, 0, color="gray", alpha=0.5, label="Messwerte")
plt.xlabel(r"Gesamtzahl der Zerfälle $N$",
           fontsize=14, family='serif')
plt.ylabel(r"Häufigkeit $p\ [\%]$",
           fontsize=14, family='serif')

X_III = np.arange(50, 140, 4)
poi = np.array([poisson(x, Pulse_ges.avr/10) * 100 for x in X_III])

#plt.clf()
rect = plt.bar((X_III*10)-100, poi*5, width=20, color="black", alpha=0.6,
               label="Poissonverteilung")
#autolabel(rect)
plt.tight_layout()
plt.legend(loc="best")
plt.show()
#print(Sum)



## Print Funktionen
