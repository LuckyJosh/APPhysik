# -*- coding: utf-8 -*-
"""
Created on Fri Nov 08 23:56:02 2013

@author: Josh

"""

from __future__ import (print_function,
                        division,
                        unicode_literals,
                        absolute_import)
import math as math
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


# Umrechenung der Termoelementspannung in Temperatur
def TensToTemp(U):
    if isinstance(U, (float, int)):
        return 25.157 * U - 0.19 * U**2

    elif all(isinstance(i, float) for i in U):
        return 25.157 * U - 0.19 * U**2

# Ermöglicht die Benutzung von ufloats als Funktionsparameter
uTensToTemp = unc.wrap(TensToTemp)


# Umrechnung von Wärmekapazität bei Konstantem Druck zu bei konstantem Volumen
def CpToCv(c, a, k, v, t):
    if isinstance(c, (float, int)):
        return c - (9 * a**2 * k * v * t)

    elif all(isinstance(i, float) for i in c):
        return c - (9 * a**2 * k * v * t)
uCpToCv = unc.wrap(CpToCv)


# Variable zu Steuerung der Ausgabe, True => Ausgabe, False => keine Ausgabe
PRINT = True
#PRINT = False

# Variabel zur Steuerung der Einheitenumformung,
# True => SI, False => keine Umformung
#SI = True
SI = False

### Laden der Messdaten


## Linearer Ausdenhungskoeffizient und Kompressionsmodul A, Q
A_CU, Q_CU, A_AL, Q_AL = np.loadtxt("Messdaten/Konstanten_Material.txt")



## Messfehler: Massen[g], Spannungen[mV]
M_ERR, U_ERR = np.loadtxt("Messdaten/Messfehler.txt")

# Umrechnung in SI-Einheiten
if SI:
    M_ERR *= 1e-03  # kg


##  Geräte Massen: Deckel_Al, Deckel_Cu, Becherglas, Kalorimeter
M_D_AL, M_D_CU, M_BG, M_KM = np.loadtxt("Messdaten/Massen_Geraete.txt")

# Umrechnung in SI-Einheiten
if SI:
    M_D_AL *= 1e-03  # kg
    M_D_CU *= 1e-03  # kg
    M_BG *= 1e-03  # kg
    M_KM *= 1e-03  # kg


## Molare Massen der Metalle M(Cu), M(Al)
mM_CU, mM_AL = np.loadtxt("Messdaten/Massen_Molar.txt")

# Umrechnung in SI-Einheiten
if SI:
    mM_CU *= 1e-03
    mM_AL *= 1e-03

## Material Massen: Al-Block+Deckel, Cu-Block+Deckel, WasserI+BG, WasserII+BG
M_AL, M_CU, M_W_1, M_W_2 = np.loadtxt("Messdaten/Massen_Material.txt")

# Umrechnung in SI-Einheiten
if SI:
    M_AL *= 1e-03  # kg
    M_CU *= 1e-03  # kg
    M_W_1 *= 1e-03  # kg
    M_W_2 *= 1e-03  # kg

# Korrektion der Werte durch Abziehen der Gerätemassen
M_AL -= M_D_AL
M_CU -= M_D_CU
M_W_1 -= M_BG
M_W_2 -= M_BG

# Errechnen des Mittelwerts der Wassermasse
M_W_avr = np.mean([M_W_1, M_W_2])

# Erstellen der Fehlerbehafteten Massen
uM_AL = ufloat(M_AL, M_ERR)
uM_CU = ufloat(M_CU, M_ERR)
uM_W_avr = ufloat(M_W_avr, M_ERR)


## Material Dichten in g/cm³: Aluminium, Kupfer, Wasser
RHO_AL, RHO_CU, RHO_W, C_W = np.loadtxt("Messdaten/Daten_Material.txt")

# Umrechnung in SI-Einheiten
if SI:
    C_W *= 1e03  # J/kgK


## Versuchsergebnisse der Materialmessung: U_c, U_h, U_m jeweils für Al & Cu
U_CU_C, U_CU_H, U_CU_M, U_AL_C, U_AL_H, U_AL_M = np.loadtxt("Messdaten/Messung_Material.txt",
                                                            unpack=True)

# Erstellen der Fehlerbehaftete Spannungen
uU_CU_C = unp.uarray(U_CU_C, len(U_CU_C)*[U_ERR])
uU_CU_H = unp.uarray(U_CU_H, len(U_CU_H)*[U_ERR])
uU_CU_M = unp.uarray(U_CU_M, len(U_CU_M)*[U_ERR])

uU_AL_C = unp.uarray(U_AL_C, len(U_AL_C)*[U_ERR])
uU_AL_H = unp.uarray(U_AL_H, len(U_AL_H)*[U_ERR])
uU_AL_M = unp.uarray(U_AL_M, len(U_AL_M)*[U_ERR])

# Erstellen von 3x3 Matrizen für die Werte von Cu und Al
# Spalte entspricht einer Versuchsreihe, Zeile: Entspricht einer Größe

uU_CU = unp.matrix([uU_CU_C, uU_CU_H, uU_CU_M])
uU_AL = unp.matrix([uU_AL_C, uU_AL_H, uU_AL_M])


# Umrechnung der Spannungen in Temperaturen

uT_CU = TensToTemp(uU_CU)
uT_AL = TensToTemp(uU_AL)

## Versuchsergebnisse der Kalorimetermessung:M_c, M_h, M_m, U_c, U_h, U_m
M_W_C, M_W_H, M_W_M, U_W_C, U_W_H, U_W_M = np.loadtxt("Messdaten/Messung_Kalorimeter.txt",
                                                      unpack=True)

# Umrechnung in SI-Einheiten
if SI:
    M_W_C *= 1e-03  # kg
    M_W_H *= 1e-03  # kg
    M_W_M *= 1e-03  # kg

# Erstellung der Fehlerhaften Wassermassen
uM_W_C = unp.uarray(M_W_C, len(M_W_C)*[M_ERR])
uM_W_H = unp.uarray(M_W_H, len(M_W_H)*[M_ERR])
uM_W_M = unp.uarray(M_W_M, len(M_W_M)*[M_ERR])

# Erstellen der Fehlerbehafteten Spannungen
uU_W_C = unp.uarray(U_W_C, len(U_W_C)*[U_ERR])
uU_W_H = unp.uarray(U_W_H, len(U_W_C)*[U_ERR])
uU_W_M = unp.uarray(U_W_M, len(U_W_C)*[U_ERR])


# Erstellen der 3x3 Messwert-Matrix
# Spalte entspricht einer Versuchsreihe, Zeile: Entspricht einer Größe
uM_W = unp.matrix([uM_W_C, uM_W_H, uM_W_M])
uU_W = unp.matrix([uU_W_C, uU_W_H, uU_W_M])


# Umrechnung der Spannungen zu Temperaturen
uT_W = TensToTemp(uU_W)


### Berechnung der Wärmekapazität des Kalorimeters
uCM_KM = unp.uarray(np.zeros(3), np.zeros(3))

##### Erinnerung: print(uU_W[Zeile, Spalte]), von 0 gezählt!


for j in range(3):
    dT_ym = uT_W[1, j] - uT_W[2, j]
    dT_mx = uT_W[2, j] - uT_W[0, j]
    cm_wy = C_W * uM_W[1, j]
    cm_wx = C_W * uM_W[0, j]
    uCM_KM[j] = ((cm_wy * dT_ym) - (cm_wx * dT_mx)) / (dT_mx)

## Mittelwert der Wärmekapazität
CM_KM_avr = np.mean(noms(uCM_KM))
CM_KM_std = np.std(noms(uCM_KM))/(len(uCM_KM)-1)
uCM_KM_avr = ufloat(CM_KM_avr, CM_KM_std)



### Berechnung der spez. Wärmekapazität der Metalle

## Aluminium
uC_AL_K = unp.uarray(np.zeros(3), np.zeros(3))

for i in range(3):
    cm_ww = C_W * uM_W_avr
    cm_gg = uCM_KM_avr 
    dT_mw = uT_AL[2, i] - uT_AL[0, i]
    dT_km = uT_AL[1, i] - uT_AL[2, i]
    uC_AL_K[i] = (cm_ww + cm_gg) * dT_mw / (uM_AL * dT_km)


## Kupfer
uC_CU_K = unp.uarray(np.zeros(3), np.zeros(3))

for i in range(3):
    cm_ww = C_W * uM_W_avr
    cm_gg = uCM_KM_avr  
    dT_mw = uT_CU[2, i] - uT_CU[0, i]
    dT_km = uT_CU[1, i] - uT_CU[2, i]
    uC_CU_K[i] = (cm_ww + cm_gg)*dT_mw/(uM_CU*dT_km)


### Berechnung der spezifischen Wärmekapazität pro Mol
uC_CU_P = uC_CU_K * mM_CU
uC_AL_P = uC_AL_K * mM_AL

TODO:
    ### Berechnung der spezifischen Wärmekapazität Cv pro Mol
    #uC_CU_V = uCpToCv(uC_CU_P, A_CU, Q_CU, )
    #uC_AL_V = uCpToCv(uC_AL_P,)


## Print Funktionen
if PRINT:
    print("\n Spannungsmatrizen:",
          "\n -Kupfer:\n", uU_CU,
          "\n\n -Aluminium:\n", uU_AL,
          "\n\n - Wasser:\n", uU_W)

    print("\n Temperaturmatrizen:",
          "\n\n -Kupfer:\n", uT_CU,
          "\n\n -Aluminium:\n", uT_AL,
          "\n\n -Wasser:\n", uT_W)

    print("\n Massenmatrix:\n",
          "\n Wasser:\n", uM_W)

    print("\n Wärmekapazität des Kalorimeters:\n", uCM_KM,
          "\n\n Mittelwert:\n", uCM_KM_avr)

    print("\n spez. Wärmekapazität pro kg:",
          "\n\n -Aluminium:\n", uC_AL_K,
          "\n\n -Kupfer: \n", uC_CU_K)

    print("\n spez. Wärmekapazität Cp pro Mol:",
          "\n\n -Aluminium:\n", uC_AL_P,
          "\n\n -Kupfer: \n", uC_CU_P)

    print("\n spez. Wärmekapazität Cv pro Mol:",
          "\n\n -Aluminium:\n", uC_AL_V,
          "\n\n -Kupfer: \n", uC_CU_V)