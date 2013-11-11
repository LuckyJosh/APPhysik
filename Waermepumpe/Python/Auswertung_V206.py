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
from uncertainties.unumpy import (nominal_values as noms,
                                  std_devs as stds)

## Mögliche Fit-Funktionen


def T_FitI(t, A, B, C):
    return A * t**2 + B * t + C


def dT_FitI(t, A, B):
    return 2 * A * t + B


def T_FitII(t, A, B, a):
    return A/(1 + B * t**a)


def T_FitIII(t, A, B, C, a):
    return ((A * t**a) / (1 + B * t**a)) + C


def P_Fit(T, A, B):
    return A * np.exp(B/T)

#%%
## Laden der Konstanten
# Laden des Zeitintervalls
DELTA_T = np.loadtxt("../Messdaten/Zeitintervall.txt")

# Laden der Füllmenge
V_H2O = np.loadtxt("../Messdaten/Fuellmenge.txt")

# Laden der Wärmekapazität
C_APP, C_SPEZ_H2O = np.loadtxt("../Messdaten/Waermekapazitaet.txt")
C_SPEZ_H2O *= 1e03  # J/K*kg
# Laden der Leistung
P_APP = np.loadtxt("../Messdaten/Leistung.txt")

# Laden der Cl2F2C ("Gas") Daten
RHO_0_GAS, K_GAS, M_MOL_GAS = np.loadtxt("../Messdaten/Transportgas.txt")

# Lade Wasserdichten
RHO_H2O_T1, RHO_H2O_T2 = np.loadtxt("../Messdaten/Wasserdichten.txt",
                                    unpack=True)

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

# 16 t- Werte in Sekunden (0, 90, 180, ..., 1350)
t = np.linspace(0, (T_dim - 1) * DELTA_T, num=T_dim)

# "Kontinuierliche" Werte von -10 bis 2000
x = np.linspace(-10, 2000, num=1000)


# Bearbeitung von T_1
    #Berechnung der Fit-Parameter
poptI_T1, pcovI_T1 = curve_fit(T_FitI, t, noms(uT_1), sigma=stds(uT_1))
poptII_T1, pcovII_T1 = curve_fit(T_FitII, t, noms(uT_1), sigma=stds(uT_1))
poptIII_T1, pcovIII_T1 = curve_fit(T_FitIII, t, noms(uT_1), sigma=stds(uT_1))

    # Berechnung der Fit-Fehler
errorI_T1 = np.sqrt(np.diag(pcovI_T1))
#errorII_T1 = np.sqrt(np.diag(pcovII_T1))
#errorIII_T1 = np.sqrt(np.diag(pcovIII_T1))

    # Plot Einstellungen
plt.clf()
plt.grid()
plt.xlabel(r"Zeit $t\,[\mathrm{s}]$")
plt.ylabel(r"Temperatur $T\,[\mathrm{K}]$")
plt.xlim(0, 1400)

    # Plot der Messpunkte
plt.errorbar(t, noms(uT_1), yerr=stds(uT_1), fmt="rx",
             label="Messwerte $T_{1}$")

    # Plots der Fit-Kurven
plt.plot(x, T_FitI(x, *poptI_T1), color="grey", 
         label="Regression-Kurve $T_{1}$")
#plt.plot(x, T_FitII(x, *poptII_T1), "k-", label="Fit2 $T_{1}$")
#plt.plot(x, T_FitIII(x, *poptIII_T1), "k-", label="Fit3 $T_{1}$")

    # Plot Legende und Speichern
plt.tight_layout()
plt.legend(loc="best")
plt.savefig("../Plots/Temperaturverlauf_T1.pdf")

# Bearbeitung von T_2
    # Berechnung der Fit-Parameter
poptI_T2, pcovI_T2 = curve_fit(T_FitI, t, noms(uT_2), sigma=stds(uT_2))
poptII_T2, pcovII_T2 = curve_fit(T_FitII, t, noms(uT_2), sigma=stds(uT_2))     # desolater Fit
poptIII_T2, pcovIII_T2 = curve_fit(T_FitIII, t, noms(uT_2), sigma=stds(uT_2))  # desolater Fit

    # Berechnung der Fit-Fehler
errorI_T2 = np.sqrt(np.diag(pcovI_T2))
#errorII_T2 = np.sqrt(np.diag(pcovII_T2))
#errorIII_T2 = np.sqrt(np.diag(pcovIII_T2))

    # Plot Einstellungen
plt.clf()
plt.grid(which="both")
plt.xlabel(r"Zeit $t\,[\mathrm{s}]$")
plt.ylabel(r"Temperatur $T\,[\mathrm{K}]$")
plt.xlim(0, 1400)
plt.ylim(275, 295)
#plt.yscale("log")
    # Plot der Messwerte
plt.errorbar(t, noms(uT_2), yerr=stds(uT_2), fmt="bx",
             label="Messwerte $T_{2}$")

    # Plots der Fit-Kurven
plt.plot(x, T_FitI(x, *poptI_T2), color="grey", 
         label="Regression-Kurve $T_{2}$")
#plt.plot(x, T_FitII(x, *poptII_T2), "k-", label="Fit2 $T_{2}$")
#plt.plot(x, T_FitIII(x, *poptIII_T2), "k-", label="Fit3 $T_{2}$")

    # Plot Legende und Speichern
plt.tight_layout()
plt.legend(loc="best")
plt.savefig("../Plots/Temperaturverlauf_T2.pdf")
#%%

## Bestimmung der Gerätegrößen (Güteziffer, Massendurchsatz & Mech. Leistung)


# 4 Werte der Ableitung T_1, für (2., 4., 6., 8.) Wert von T_1
dT_1 = np.zeros(4)
for i in range(4):
    dT_1[i] = dT_FitI(t[(i+1)*2], poptI_T1[0], poptI_T1[1])

# 4 Werte der Ableitung T_2, für (2., 4., 6., 8.) Wert von T_2
dT_2 = np.zeros(4)
for i in range(4):
    dT_2[i] = dT_FitI(t[(i+1)*2], poptI_T2[0], poptI_T2[1])


    # Güteziffer
Nu_id = np.zeros(4)
for i in range(4):
    Nu_id[i] = (noms(uT_1[(i + 1) * 2])/(noms(uT_1[(i + 1) * 2])
                - noms(uT_2[(i + 1) * 2])))

M_H2O_T1 = V_H2O * RHO_H2O_T1[0]

C_H2O_T1 = C_SPEZ_H2O * M_H2O_T1

dQ_1 = np.zeros(len(dT_1))
for i in range(len(dT_1)):
    dQ_1[i] = (C_H2O_T1 + C_APP) * dT_1[i]


Nu_real = dQ_1/P_APP


    # Massendurchsatz


M_H2O_T2 = V_H2O * RHO_H2O_T2[0]

C_H2O_T2 = C_SPEZ_H2O * M_H2O_T2
print(C_H2O_T2)

dQ_2 = np.zeros(len(dT_2))
for i in range(len(dT_2)):
    dQ_2[i] = (C_H2O_T2 + C_APP) * dT_2[i]

        # Regerssion von (P_2, T_2)

popt_P, pcov_P = curve_fit(P_Fit, noms(uT_1), noms(uP_1))
error_P = np.sqrt(np.diag(pcov_P))


plt.clf()
plt.grid(which="both")
plt.xlabel(r"Temperatur $T\,[\mathrm{K}]$")
plt.ylabel(r"Druck $p\,[\mathrm{Pa}]$")
plt.yscale("log")

plt.plot(1/noms(uT_1), noms(uP_1), "rx", label="Messwerte")
plt.plot(1/noms(uT_1), P_Fit(noms(uT_1), *popt_P), "k-", label="Messwerte")

plt.show()

L = popt_P[1] * (const.gas_constant) * (-1)

print(L)
dm_mol = dQ_2 / L
dm = dm_mol * M_MOL_GAS


    # Kompressorleistung
        # allgemeine Gasgleichung p_0 * V/T_0 = nR
rho_gas = RHO_0_GAS  * 273.15 * P_2 /( 1e05 * T_2)
print(rho_gas)

def aux(var):
    return (var + 1) * 2


N_App = np.zeros(4)
X = 1/(K_GAS - 1)
x = (1/K_GAS)
for j in range(4):
    P_b = P_1[aux(j)]
    P_a = P_2[aux(j)]
    W_P = (P_a/P_b)**x
    K = P_a - P_b * W_P
    N_App[j] = X * (dm[j]/rho_gas[aux(j)]) * (K)









## Ausgaben
print("Temperaturen:\n", "-Warm:\n", uT_1, "\n", "-Kalt:\n", uT_2)
print("Drücke:\n", "-Warm:\n", uP_1, "\n", "-Kalt:\n", uP_2)
print("Zeiten:\n", t)
print("Parameter:\n", "Fit1:\n", poptI_T1, "\n", "Fit2:\n", poptII_T1, "\n",
      "Fit3:\n", poptIII_T1)
print("Parameter:\n", "Fit1:\n", poptI_T2, "\n", "Fit2:\n", poptII_T2, "\n",
      "Fit3:\n", poptIII_T2)
print("Fehler:\n", "T_1:\n", errorI_T1, "\n", "T_2:\n", errorI_T2)
print("Ableitungen von T_1:\n", dT_1, "\n", "Ableitungen von T_2:\n", dT_2)
print("Güteziffer:\n", "-ideal:\n", Nu_id, "\n", "-real:\n", Nu_real)
#print("Masse Wasser:\n", M_H2O)
#print("Wärmekapazität Wasser:\n", C_H2O)
#print("Wärmeänderung dQ_1:\n", dQ_1)
print("Wärmeänderung dQ_2:\n", dQ_2)
print("Fit-Parameter P-Fit:\n", *popt_P)
print("Massendurchsatz:\n", dm)
print("Mechanische Leistung:\n", N_App)
