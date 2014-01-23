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
from uncertainties.unumpy import uarray
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)

sys.path.append("..\_globales\python")
import latextables as lxtabs

### Uncertainties Funktionen
Umean = unc.wrap(np.mean)

###


def gearToIndex(g):
    return m.trunc((int(g) / 6) - 1)


"""

Begin der Auswertung zum Dopplereffekt

"""

# Laden der Daten zur Bestimmung der Geschwindigkeit
## Laden der Strecke
l = np.loadtxt("Messdaten/Strecke.txt", unpack=True)
l_err = np.loadtxt("Messdaten/Fehler_Strecke.txt")

### Fehler behaftete Messwerte
ul = unp.uarray(l, [l_err]*len(l))

### Mittelwert
ul_avr = Umean(ul)
ul_avr = ufloat(noms(uL), stds(uL))
ul_avr *= 1e-02  # [cm] --> [m]

## Laden der Zeiten in den verschiedenen Gängen
G, t_h1, t_h2, t_r1, t_r2 = np.loadtxt("Messdaten/Zeiten.txt", unpack=True)
t_err = np.loadtxt("Messdaten/Fehler_Zeiten.txt")

### Fehlerbehaftete Messwerte
ut_h1 = unp.uarray(t_h1, [t_err]*len(t_h1))
ut_h2 = unp.uarray(t_h2, [t_err]*len(t_h2))
ut_r1 = unp.uarray(t_r1, [t_err]*len(t_r1))
ut_r2 = unp.uarray(t_r2, [t_err]*len(t_r2))

### Mittelwerte der Zeiten

uT_h_avr = unp.uarray(np.zeros(len(G)), np.zeros(len(G)))
for g in G:
    uT_h_avr[gearToIndex(g)] = Umean([ut_h1[gearToIndex(g)],
                                      ut_h2[gearToIndex(g)]])
    uT_h_avr[gearToIndex(g)] = (ufloat(noms(uT_h_avr[gearToIndex(g)]),
                                       stds(uT_h_avr[gearToIndex(g)])))

uT_r_avr = unp.uarray(np.zeros(len(G)), np.zeros(len(G)))
for g in G:
    uT_r_avr[gearToIndex(g)] = Umean([ut_r1[gearToIndex(g)],
                                      ut_r2[gearToIndex(g)]])
    uT_r_avr[gearToIndex(g)] = (ufloat(noms(uT_r_avr[gearToIndex(g)]),
                                       stds(uT_r_avr[gearToIndex(g)])))

## Berechnung der Geschwindigkeiten
uv_h = ul_avr / uT_h_avr
uv_r = ul_avr / uT_r_avr




# Bestimmung der Schalgeschwindigkeit in Luft bei Raumtemperatur
## Bestimmen der Grundfrequenz f0
### Laden der Grundfrequenzen
f0 = np.loadtxt("Messdaten/Ruhefrequenzen.txt", unpack=True)

#### Bestimmung des Mittlewerts
f0_avr = np.mean(f)
##### Abweichung vom Mittelwert
f0_std = np.std(f)/(np.sqrt(len(f)))

##### Fehlerbeahfteter Mittelwert
uf0_avr = ufloat(f0_avr, f0_std)


## Bestimmung der Wellenlänge
### Laden der Messdaten
s0, s1, k = np.loadtxt("Messdaten/Wellenlaengen.txt", unpack=True)
s_err = np.loadtxt("Messdaten/Fehler_Wellenlaengen.txt")

#### Fehlerbehaftete Messwerte
us0 = uarray(s0, s_err)
us1 = uarray(s1, s_err)

us0 *= 1e-02  # [cm]-->[m]
us1 *= 1e-02  # [cm]-->[m]

### Berechnung der Wellenlänge
uk = (us0 + us1)/k


## Berechnung der Schallgeschwindigkeit
uc = uk * uf0_avr

### Bestimmung des Mittlewertes
uc_avr = Umean(uC)
#### Abweichung vom Mittelwert
c_std = np.std(noms(uc))/np.sqrt(len(uC))

uc_avr = ufloat(noms(uc_avr), c_std)


# Messung der Frequenz
## Laden der Frequenzen in den verschiedenen Gängen
G, f_h1, f_h2, f_r1, f_r2 = np.loadtxt("Messdaten/Frequenzmessung.txt",
                                       unpack=True)
f_err = np.loadtxt("Messdaten/Fehler_Frequenz.txt")

### Fehlerbehaftete Messwerte
uf_h1 = unp.uarray(f_h1, [f_err]*len(f_h1))
uf_h2 = unp.uarray(f_h2, [f_err]*len(f_h2))
uf_r1 = unp.uarray(f_r1, [f_err]*len(f_r1))
uf_r2 = unp.uarray(f_r2, [f_err]*len(f_r2))

### Mittelwerte der Zeiten

uF_h_avr = unp.uarray(np.zeros(len(G)), np.zeros(len(G)))
for g in G:
    uF_h_avr[gearToIndex(g)] = Umean([uf_h1[gearToIndex(g)],
                                      uf_h2[gearToIndex(g)]])
    uF_h_avr[gearToIndex(g)] = (ufloat(noms(uF_h_avr[gearToIndex(g)]),
                                       stds(uF_h_avr[gearToIndex(g)])))

uF_r_avr = unp.uarray(np.zeros(len(G)), np.zeros(len(G)))
for g in G:
    uF_r_avr[gearToIndex(g)] = Umean([uf_r1[gearToIndex(g)],
                                      uf_r2[gearToIndex(g)]])
    uF_r_avr[gearToIndex(g)] = (ufloat(noms(uF_r_avr[gearToIndex(g)]),
                                       stds(uF_r_avr[gearToIndex(g)])))


### Bestimmung der Frequenzänderung
udf_h = uF_h_avr - uf0_avr
udf_r = uF_r_avr - uf0_avr


# Messung der  Frequenzänderung durch Schwebung

## Laden der Frequenzen in den verschiedenen Gängen
G, df_h1, df_h2, df_r1, df_r2 = np.loadtxt("Messdaten/Schwebung.txt",
                                       unpack=True)
df_err = np.loadtxt("Messdaten/Fehler_Frequenz.txt")

### Fehlerbehaftete Messwerte
udf_h1 = unp.uarray(df_h1, [df_err]*len(df_h1))
udf_h2 = unp.uarray(df_h2, [df_err]*len(df_h2))
udf_r1 = unp.uarray(df_r1, [df_err]*len(df_r1))
udf_r2 = unp.uarray(df_r2, [df_err]*len(df_r2))

### Mittelwerte der Zeiten

udF_h_avr = unp.uarray(np.zeros(len(G)), np.zeros(len(G)))
for g in G:
    udF_h_avr[gearToIndex(g)] = Umean([udf_h1[gearToIndex(g)],
                                      udf_h2[gearToIndex(g)]])
    udF_h_avr[gearToIndex(g)] = (ufloat(noms(udF_h_avr[gearToIndex(g)]),
                                        stds(udF_h_avr[gearToIndex(g)])))

udF_r_avr = unp.uarray(np.zeros(len(G)), np.zeros(len(G)))
for g in G:
    udF_r_avr[gearToIndex(g)] = Umean([udf_r1[gearToIndex(g)],
                                      udf_r2[gearToIndex(g)]])
    udF_r_avr[gearToIndex(g)] = (ufloat(noms(udF_r_avr[gearToIndex(g)]),
                                        stds(udF_r_avr[gearToIndex(g)])))






## Print Funktionen










