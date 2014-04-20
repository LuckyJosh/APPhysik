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
from aputils.utils import Quantity, ErrorEquation


def func(x, a, b):
    return a*x + b

def effectiveLength(p, x_0, p_0 = 1013):
    return x_0 * (p / p_0)

def distanceToEnergy(R):
    return (R/(3.1))**(2/3)

PRINT=True
SHOW = False

###########################################################################
#
# Aufgabenteil I
#
###########################################################################

#Berechnung der Energie der Channels


def channelToEnergy(ch, ch_list):
    channel_max = max(ch_list)
    channel_min = 0
    energy_slope = 4/(channel_max - channel_min)
    energy_intercept = 4 - (channel_max*energy_slope)
    return energy_slope * ch + energy_intercept


#Laden der Messdaten
p_I_all, ch_I_all, pulse_I_all = np.loadtxt("Messdaten/MessreiheI.txt", unpack=True)
p_I = np.delete(p_I_all, 3)
ch_I = np.delete(ch_I_all, 3)
pulse_I = np.delete(pulse_I_all, 3)


#Berechnung der effektiven Längen
x_eff = effectiveLength(p_I, 20)
#Berechnung der Zerfallsrate
rate_I = pulse_I/120
#Bestimmung der halben maximal Rate
rate_max_half_I = max(rate_I)/2
#Bestimmung der Energien der Channels
energy_I_all = [channelToEnergy(ch, ch_I) for ch in ch_I_all]
energy_I = [channelToEnergy(ch, ch_I) for ch in ch_I]

# Tabelle der Messdaten
T_I = Table(siunitx=True)
T_I.layout(seperator="column", title_row_seperator="double", border="true")
T_I.caption(r"Messwerte der Messung im Abstand von $20 \si{mm}$")
T_I.label("Messwerte_I")
T_I.addColumn([int(p) for p in p_I_all], title="Druck", symbol="p",
              unit=r"\milli\bar")
T_I.addColumn([int(ch) for ch in ch_I_all], title="Channel Maximum",
              symbol="Ch_{max}")
T_I.addColumn(energy_I_all, title="Energie Maximum", symbol="E_{max}",
              unit=r"\mega\eV")
T_I.addColumn([int(p) for p in pulse_I_all], title="Anzahl Pulse", symbol="N")
T_I.show() if SHOW else T_I.save("Tabellen/Messwerte_I.tex")


#Berechnung der Fit-Parameter (gerade)
popt, pcov = curve_fit(func, x_eff[0:-7], rate_I[0:-7])
#Bestimmung der Parameter-Fehler
errors = np.sqrt(np.diag(pcov))
#Fehlerbehaftete Parameter
param_a_I = ufloat(popt[0], errors[0])
param_b_I = ufloat(popt[1], errors[1])
print("FitparameterI:", param_a_I, param_b_I)

# Bestimmung des Schnittpunktes (mittlere Reichweite/Halbe maximal Rate)
X = np.linspace(-10, 20, 100000)

f = [func(x, popt[0], popt[1]) for x in X]
lower = []
greater = []
for i in f:
    if i < rate_max_half_I:
        lower.append(i)
    if i > rate_max_half_I:
        greater.append(i)
intercept_y = (min(greater), max(lower))
intercept_x = X[np.where(f == min(greater))[0][0]]
print("Schnittpunkt x / Mittlere Reichweite:", intercept_x)
print("Schnittpunkt y:", rate_max_half_I, intercept_y)

#Bestimmung der Energie, ausgehend von der mittleren Reichweite
distance_avr_I = intercept_x
#energy_I = distanceToEnergy(distance_avr_I)
energy_I = distanceToEnergy(ufloat(distance_avr_I, 0.01))
print("Energie der Alphateilchen:", energy_I)

# Erstellen des Plots
plt.clf()
plt.xlim(-5, 20)
plt.ylim(0, 300)
plt.grid()
plt.xlabel("Effektive Länge $x\ [\mathrm{mm}]$", fontsize=14, family='serif')
plt.ylabel("Zerfallsrate $A\ [\mathrm{s^{-1}}]$", fontsize=14, family='serif')

#Messwerte für Regression
plt.plot(x_eff[0:-7], rate_I[0:-7], "rx", label="Messwerte")
#Messwerte ohne Regression
plt.plot(x_eff[-7:], rate_I[-7:], "kx")
#Fit
plt.plot(X, func(X, popt[0], popt[1]), label="Regressionsgerade", color="gray")
#Halbesmaximum
plt.plot(X, len(X)*[rate_max_half_I],"r--", alpha=0.7, label="Halbe maximal Zerfallsrate")
#Schnittpunkt
plt.plot(intercept_x, rate_max_half_I, "ko", alpha=0.7,
         label="Schnittpunkt ({}|{})".format(round(intercept_x, 2),
                                             round(rate_max_half_I, 2)))

plt.tight_layout()
plt.legend(loc="best")
plt.show() if SHOW else plt.savefig("Grafiken/MittlereReichweiteI.pdf")




###########################################################################
#
# Aufgabenteil II
#
###########################################################################

#Laden der Messdaten
p_II, ch_II, pulse_II = np.loadtxt("Messdaten/MessreiheII.txt", unpack=True)
#Bestimmung der effektiven Länge
x_eff_II = effectiveLength(p_I, 25)
#Bestimmung der Zerfallsrate
rate_II = pulse_I/120
#Bestimmung der halben maximal Rate
rate_max_half_II = max(rate_II)/2
# Berechnung der Energie der Channels
energy_II = channelToEnergy(ch_II, ch_II)

# Tabelle  der Messwerte
T_II = Table(siunitx = True)
T_II.layout(seperator="column", title_row_seperator="double", border="true")
T_II.caption(r"Messwerte der Messung im Abstand von $25 \si{mm}$")
T_II.label("Messwerte_II")
T_II.addColumn([int(p) for p in p_II], title="Druck", symbol="p", unit=r"\milli\bar")
T_II.addColumn([int(ch) for ch in ch_II], title="Channel Maximum",
                symbol="Ch_{max}")
T_II.addColumn(energy_II, title="Energie Maximum", symbol="E_{max}", unit=r"\mega\eV")
T_II.addColumn([int(p) for p in pulse_II], title="Anzahl Pulse", symbol="N")
T_II.show() if SHOW else T_II.save("Tabellen/Messwerte_II.tex")





popt, pcov = curve_fit(func, x_eff_II[0:-5], rate_II[0:-5])
errors = np.sqrt(np.diag(pcov))
param_a_II = ufloat(popt[0], errors[0])
param_b_II = ufloat(popt[1], errors[1])
print("FitparameterII:", param_a_II, param_b_II)

#Bestimmung des Schnittpunktes(mittlere Reichweite/halbe maximal Rate)

X = np.linspace(-10, 25, 100000)
f = [func(x, popt[0], popt[1]) for x in X]
lower = []
greater = []
for i in f:
    if i < rate_max_half_II:
        lower.append(i)
    if i > rate_max_half_II:
        greater.append(i)
intercept_y = (min(greater), max(lower))
intercept_x = X[np.where(f == min(greater))[0][0]]
print("Schnittpunkt x / Mittlere Reichweite:", intercept_x)
print("Schnittpunkt y:", rate_max_half_II, intercept_y)

#Bestimmung der Energie aus der mittleren Reichweite
distance_avr_I = intercept_x
energy_I = distanceToEnergy(distance_avr_I)
energy_I = distanceToEnergy(ufloat(distance_avr_I, 0.01))
print("Energie der Alphateilchen:", energy_I)

#Erstellen des zweiten Plots
plt.clf()
plt.xlim(-5, 25)
plt.ylim(0, 300)
plt.grid()
plt.xlabel("Effektive Länge $x\ [\mathrm{mm}]$", fontsize=14, family='serif')
plt.ylabel("Zerfallsrate $A\ [\mathrm{s^{-1}}]$", fontsize=14, family='serif')
#Messwerte mit Regression
plt.plot(x_eff_II[0:-5], rate_II[0:-5], "rx", label="Messwerte")
#Messwerte ohne Regression
plt.plot(x_eff_II[-5:], rate_II[-5:], "kx")
#Fit
plt.plot(X, func(X, popt[0], popt[1]), label="Regressionsgerade", color="gray")
#Halbe maximal Rate
plt.plot(X, len(X)*[rate_max_half_II], "r--", alpha=0.7,
         label="Halbe maximal Zerfallsrate")
#Schnittpunkt
plt.plot(intercept_x, rate_max_half_II, "ko", alpha=0.7,
         label="Schnittpunkt ({}|{})".format(round(intercept_x, 2),
                                             round(rate_max_half_II, 2)))

plt.tight_layout()
plt.legend(loc="best")
plt.show() if SHOW else plt.savefig("Grafiken/MittlereReichweiteII.pdf")


# Bestimmung der Energieänderung für II statt für I



def func_II(x, a, b):
    return - a * np.log(b * x)
popt, pcov = curve_fit(func_II, effectiveLength(p_II, 25)[1:],
                       channelToEnergy(ch_II, ch_II)[1:])
error = np.sqrt(np.diag(pcov))
param_a_III = ufloat(popt[0], error[0])
param_b_III = ufloat(popt[1], error[1])

print("Logarithmuns Parameter:", param_a_III, param_b_III)


eff_length = np.linspace(0.1, 25, 500)

plt.clf()
plt.grid()
plt.xlim(-1, 25)
plt.ylim(2.75, 4.2)

plt.plot(effectiveLength(p_II, 25), channelToEnergy(ch_II, ch_II), "rx",
         label="Messwerte")
plt.plot(eff_length, func_II(eff_length, popt[0], popt[1]),
         label="Regressionskurve")
plt.xlabel("Effektive Länge $x\ [\mathrm{mm}]$", fontsize=14, family='serif')
plt.ylabel("maximal Energie $E_{max}\ [\mathrm{MeV}]$",
           fontsize=14, family='serif')
plt.tight_layout()
plt.legend(loc="best")
plt.show() if SHOW else plt.savefig("Grafiken/EnergieVerlauf.pdf")


#####################################
#
# Aufgabenteil III
#
#####################################

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2., 1.0*height, '%d'%int(height),
                 ha='center', va='bottom', size=10)

def poisson(x, mu):
    return mu**x * np.exp(-mu)/m.factorial(x)


#Bestimmung der der mittleren Zerfallsrate
pulse = np.loadtxt("Messdaten/MessreiheIII.txt")
pulse_III = pulse
pulse /= 10
Pulse_ges = Quantity(pulse)
print("Mittelwert, Abweichung:",Pulse_ges.avr, Pulse_ges.std, Pulse_ges.avr_err)


# Tabelle  der Messwerte
#t = range(101)[1:]
#T_III = Table(siunitx = True)
#T_III.layout(seperator="column", title_row_seperator="double", border="true")
#T_III.caption(r"Anzahl der gemessenen Impulse")
#T_III.label("Messwerte_III")
#T_III.addColumn(t[:], title="Versuch Nr.")
#T_III.addColumn([int(p) for p in pulse_III[:]], title="Anzahl Pulse", symbol="N")
#T_III.show() if SHOW else T_III.save("Tabellen/Messwerte_III.tex")



# Sortierung der Messergebnisse in N balken der Breite dN
ranges = np.arange(500, 1400, 20)
Lists = []
#print(min(pulse), max(pulse), max(pulse) - min(pulse))
for i in range(len(ranges)):
    if i == len(ranges) -1:
        break
    List = []
    for p in pulse:
        if ranges[i] < p < ranges[i + 1]:
            List.append(p)
    Lists.append(List)

#for l in Lists:
#    print(l, len(l))

# Ploten der Messewerte in einem Histogramm
plt.clf()
Sum = 0
for j in range(len(ranges)):
    if not j == len(ranges) - 1 and not len(Lists[j]) == 0:
        Sum += len(Lists[j])
        rect = plt.bar(ranges[j], len(Lists[j]),
                       width=20, color="red", alpha=0.7)
        #autolabel(rect)
else:
    plt.bar(1400, 0, color="red", alpha=0.7, label="Messwerte")
    #plt.bar(Pulse_ges.avr, 6, width=5, color="blue", label="Mittelwert")
    plt.ylim(0, 18)
    plt.xlabel(r"Zerfallsrate $A \ [\mathrm{s^{-1}}] $",
               fontsize=14, family='serif')
    plt.ylabel(r"Häufigkeit $P\ [\mathrm{\%}] $",
               fontsize=14, family='serif')
    plt.grid(axis="y")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show() if SHOW else plt.savefig("Grafiken/AktivitaetHistogramm.pdf")
print("Summe der Messungen:", Sum)


#Erstellen des Plots Poisson & Messwerte
# Erstellen einer Poissonverteilung
X_III = np.arange(0, 140, 2)
poi = np.array([poisson(x, Pulse_ges.avr/10) * 100 for x in X_III])

plt.clf()

for j in range(len(ranges)):
    if not j == len(ranges) - 1 and not len(Lists[j]) == 0:
        Sum += len(Lists[j])
        rect = plt.bar(ranges[j], len(Lists[j]),
                       width=10, color="red", alpha=0.7)#, label="Messwerte")
else:
    plt.bar(1400, 0, color="red", alpha=0.7, label="Messwerte")


rect = plt.bar((X_III*10)-10, poi*3.5, width=10, color="blue", alpha=0.6,
               label="Poissonverteilung")
plt.xlim(350, 1300)
plt.xlabel(r"Zerfallsrate $A \ [\mathrm{s^{-1}}] $",
           fontsize=14, family='serif')
plt.ylabel(r"Häufigkeit $P \ [\%]$",
           fontsize=14, family='serif')
plt.tight_layout()
plt.legend(loc="best")
plt.grid(axis="y")
plt.show() if SHOW else plt.savefig("Grafiken/VergleichPoisson.pdf")


#Erstellen der Gaussverteilung
def gauss(x, mu, sig):
    return 1/(sqrt(2*const.pi)*sig) * np.exp(- (x - mu)**2/(2 * sig**2))
X_IV = np.arange(0, 140, 0.2)
gau = np.array([gauss(x, Pulse_ges.avr/10, Pulse_ges.std/10) for x in X_IV])

plt.clf()

for j in range(len(ranges)):
    if not j == len(ranges) - 1 and not len(Lists[j]) == 0:
        Sum += len(Lists[j])
        rect = plt.bar(ranges[j], len(Lists[j]),
                       width=10, color="red", alpha=0.7)#, label="Messwerte")
else:
    plt.bar(1400, 0, color="red", alpha=0.7, label="Messwerte")


#rect = plt.bar((X_IV*10)-10, gau*500, width=10, color="blue", alpha=0.6,
#               label="Gaussverteilung")
plt.plot((X_IV*10)-10, gau*500, "b-", label="Gaussverteilung")

plt.xlim(350, 1300)
plt.ylim(0, 18)
plt.xlabel(r"Zerfallsrate $A \ [\mathrm{s^{-1}}] $",
           fontsize=14, family='serif')
plt.ylabel(r"Häufigkeit $P\ [\%]$",
           fontsize=14, family='serif')
plt.tight_layout()
plt.legend(loc="best")
plt.grid(axis="y")
plt.show() if SHOW else plt.savefig("Grafiken/VergleichGauss.pdf")
## Print Funktionen

## Fehler
R = var("R_m")
EnergieError = ErrorEquation((R/3.1)**(2/3), name="E")
print(EnergieError.std)
