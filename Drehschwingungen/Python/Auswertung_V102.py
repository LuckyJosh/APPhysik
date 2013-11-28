# -*- coding: utf-8 -*-
"""
Created on Sat Nov 02 22:00:59 2013

@author: Josh
"""

from __future__ import (print_function,
                        division,
                        unicode_literals,
                        absolute_import)
import math as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit
import sympy as sp
from uncertainties import ufloat
import uncertainties.unumpy as unp



## Trosionsmoduls G
 
 ## Bestimmung des Trägheitsmoments der Kugel I_K
def I_Kugel(m, r):
    return 0.4 * m * r**2

  # Lade Kugelmasse und -durchmesser (+Fehler)
  # und Trägheitsmoment der Halterung
M_K, M_K_err, D_K, D_K_err, I_H = np.loadtxt("../Messdaten/Dimension_Kugel.txt",
                                             unpack=True)
  # SI-Einheiten
M_K *= 1e-03      # kg
M_K_err *= 1e-02  # Prozent
D_K *= 1e-03      # m
D_K_err *= 1e-02  # Prozent
I_H *= 1e-07      # kgm²

  # Kugelmasse mit Fehler
uM_K = ufloat(M_K, M_K * M_K_err)  # kg

  # Kugeldurchmesser mit Fehler
uD_K = ufloat(D_K, D_K * D_K_err)  # m

  # Kugelträgheitsmoment mit Fehler
I_K = I_Kugel(uM_K, (uD_K/2))
uI_K = ufloat(unp.nominal_values(I_K), unp.std_devs(I_K))  # kgm²

  # Gesamtträgheitsmoment mit Fehler
I_ges = uI_K + I_H
uI_ges = ufloat(unp.nominal_values(I_ges), unp.std_devs(I_ges))  # kgm²
print("Trägheitsmoment Kugel:", uI_K)
print("Trägheitsmoment gesamt:", uI_ges)



 ## Bestimmung der gemittelten Periodendauer uT_avr

  # Lade Periodendauer ohne Magnetfeld
T = np.loadtxt("../Messdaten/Periodendauer_ohne_Magnetfeld.txt", unpack=True)

  # Lade Fehler der Zeitmessung
T_err = np.loadtxt("../Messdaten/Periodendauer_Fehler.txt")

  # Periodendauer mit Fehler
uT = unp.uarray(T, len(T)*[T_err])

  # gemittelte Periodendauer mit Fehler
uT_avr = np.mean(uT)
T_avr = np.mean(T)
T_std = np.std(T)/(len(T))
UT_avr = ufloat(T_avr, T_std)
uT_avr = ufloat(unp.nominal_values(uT_avr), unp.std_devs(uT_avr))
print("Mittlere Periodendauer ohne B:", uT_avr, UT_avr)
 ## Verarbeitung der Drahtdaten

  # Lade Drahtdurchmesser
D_D = np.loadtxt("../Messdaten/Dimension_Draht_Durchmesser.txt", unpack=True)

  # Lade Drahtdurchmesser Fehler
D_D_err = np.loadtxt("../Messdaten/Dimension_Draht_Durchmesser_Fehler.txt")

  # SI -Einheiten
D_D *= 1e-03      # m
D_D_err *= 1e-03  # m

  # Drahtdurchmesser mit Fehler
uD_D = unp.uarray(D_D, len(D_D)*[D_D_err])

  # gemittelter Drahtdurchmesser mit Fehler
uD_D_avr = np.mean(uD_D)
  # abweichung des Mittelwertes
uD_D_avr = ufloat(unp.nominal_values(uD_D_avr), unp.std_devs(uD_D_avr))  # m
uR_D_avr = uD_D_avr / 2
uR_D_avr = ufloat(unp.nominal_values(uR_D_avr), unp.std_devs(uR_D_avr))
print("Mittlerer Drahtdurchmesser:", uD_D_avr)
print("Mittlerer Drahtradius:", uR_D_avr)

  # Lade Drahtlänge(+ Fehler)
L_D, L_D_err = np.loadtxt("../Messdaten/Dimension_Draht_Laenge.txt",
                          unpack=True)

  # SI-Einheiten
L_D *= 1e-02      # m
L_D_err *= 1e-02  # m

  # Drahtlänge mit Fehler
uL_D = ufloat(L_D, L_D_err)  # m


 ## Bestimmung des Torsionsmoduls uG
uG = (8 * const.pi * uL_D)/(uT_avr**2 * (uR_D_avr)**4) * uI_ges
print("Torsionsmodul G:", uG)


## Verarbeitung des Elastizitätsmoduls

 # Lade Elastizitätsmodul(+Fehler)
E, E_err = np.loadtxt("../Messdaten/Elastizitaetsmodul.txt", unpack=True)

 # Elastizitätsmodul mit Fehler
uE = ufloat(E, E_err)
print("Elastizitätsmodul E:", uE)

## Bestimmung der Poissonschen Querkontraktionszahl uP
## und des Kompressionsmoduls uQ

P = (uE / (2 * uG)) - 1
uP = ufloat(unp.nominal_values(P), unp.std_devs(P))
print("Querkontraktionszahl P:", uP)

Q = uE / (3 * (1 - (2 * uP)))
uQ = ufloat(unp.nominal_values(Q), unp.std_devs(Q))
print("Kompressionsmodul Q:", uQ)


## magenetisches Moment

 ## Magnetfeld einer Helmoltzspule ("Das Physikalische Praktikum")
def helmholtz(n, i, r):
    return const.mu_0 * (8/ma.sqrt(125)) * (n * i) / r


  # Lade Spulenradius, -windungszahl und -strom(+Fehler)
R, N, I, I_err = np.loadtxt("../Messdaten/Dimension_Spule.txt", unpack=True)

  # SI - Einheiten
R *= 1e-03  # m
I_err *= 1e-02  # Prozent

  # Spulenstrom mit Fehler
uI = ufloat(I, I * I_err)  # A
print("Spulenstrom I:", uI)
  # Spulenmagentfeld mit Fehler
B = helmholtz(N, uI, R)
uB = ufloat(unp.nominal_values(B), unp.std_devs(B))  # T
print("Magnetfeld B:", uB)

 ## Verarbeitung der Periodendauer
  # Lade Periodendauer T_m
T_m = np.loadtxt("../Messdaten/Periodendauer_mit_Magnetfeld.txt",
                 unpack=True)

  # Lade Periodendauerfehler
T_m_err = np.loadtxt("../Messdaten/Periodendauer_Fehler.txt")

  # Periodendauer mit Fehler
uT_m = unp.uarray(T_m, len(T_m)*[T_m_err])

  # gemittelte Periodendauer mit Fehler
T_m_avr = np.mean(T_m)
T_m_std = np.std(T_m)/(len(T_m)-1)
UT_m_avr = ufloat(T_m_avr, T_m_std)
uT_m_avr = np.mean(uT_m)
uT_m_avr = ufloat(unp.nominal_values(uT_m_avr), unp.std_devs(uT_m_avr))
print("Mittler Periodendauer mit B:", uT_m_avr, UT_m_avr)
  ## Berechnung der magnetischen Moments um
um = (4 * const.pi**2 * (uI_ges)/(uB * uT_m_avr**2) -
     ((const.pi * uG * (uR_D_avr)**4)/(2 * uL_D*uB)))
print("Magnetisches Moment m:", um)



