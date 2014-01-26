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
ul_avr = ufloat(noms(ul_avr), stds(ul_avr))
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

## alle Geschwindigkeiten mit Vorzeichen
uv = np.concatenate((uv_h, -uv_r))


# Bestimmung der Schalgeschwindigkeit in Luft bei Raumtemperatur
## Bestimmen der Grundfrequenz f0
### Laden der Grundfrequenzen
f0 = np.loadtxt("Messdaten/Ruhefrequenzen.txt", unpack=True)
f_err = np.loadtxt("Messdaten/Fehler_Frequenz.txt")

uf0 = uarray(f0, [f_err]*len(f0))

#### Bestimmung des Mittlewerts
f0_avr = np.mean(f0)
##### Abweichung vom Mittelwert
f0_std = np.std(f0)/(np.sqrt(len(f0)))

##### Fehlerbeahfteter Mittelwert
uf0_avr = ufloat(f0_avr, f0_std)


## Bestimmung der Wellenlänge
### Laden der Messdaten
s0, s1, n = np.loadtxt("Messdaten/Wellenlaengen.txt", unpack=True)
s_err = np.loadtxt("Messdaten/Fehler_Wellenlaengen.txt")

#### Fehlerbehaftete Messwerte
us0 = uarray(s0, s_err)
us1 = uarray(s1, s_err)

us0 *= 1e-02  # [cm]-->[m]
us1 *= 1e-02  # [cm]-->[m]
us = us1 + us0


### Berechnung der Wellenlänge
uk = (us0 + us1)/n
uk_avr = Umean(uk)
uk_inv = 1/uk

uk_inv_avr = Umean(uk_inv)
k_inv_std = np.std(noms(uk_inv))/np.sqrt(len(uk_inv))
uk_inv_avr = ufloat(noms(uk_inv_avr), k_inv_std)

## Berechnung der Schallgeschwindigkeit
uc = uk * uf0_avr
#uC 1/uk_inv_avr * uf0_avr  # Für den Bericht verwedet

### Bestimmung des Mittlewertes
uc_avr = Umean(uc)
#### Abweichung vom Mittelwert
c_std = np.std(noms(uc))/np.sqrt(len(uc))

uc_avr = ufloat(noms(uc_avr), c_std)

#### Berechnung der Inversenwellenlänge
#uk_inv_avr = uf0_avr/uc_avr

# Abweichung von (2) und (5)
d_gln = uf0_avr * (max(uv_h)/uc_avr)**2


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

#### alle Frequenzänderungen mit Vorzeichen
udf = np.concatenate((udf_h, udf_r))

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
#### Alle Frequenzänderungen mit Voreichen
udF = np.concatenate((udF_h_avr, -udF_r_avr))

# Grafische Auswertung

def y(x, m, b):
    return m * x + b
def y1(x, m):
    return m * x

popt1, pcov1 = curve_fit(y, noms(uv), noms(udf), sigma=stds(udf))
errors1 = np.sqrt(np.diag(pcov1))
params1 = uarray(popt1, errors1)

popt15, pcov15 = curve_fit(y1, noms(uv), noms(udf), sigma=stds(udf))
errors15 = np.sqrt(np.diag(pcov15))
params15 = uarray(popt15, errors15)

udF_plot = np.concatenate((udF[1:-10], udF[11:]))
uv_plot = np.concatenate((uv[1:10], uv[11:]))
popt25, pcov25 = curve_fit(y1, noms(uv_plot), noms(udF_plot),
                         sigma=stds(udF_plot))
errors25 = np.sqrt(np.diag(pcov25))
params25 = uarray(popt25, errors25)
popt2, pcov2 = curve_fit(y, noms(uv_plot), noms(udF_plot),
                         sigma=stds(udF_plot))
errors2 = np.sqrt(np.diag(pcov2))
params2 = uarray(popt2, errors2)

V = np.linspace(-50, 50, num=200)
## Erstellen des Plots
### Auftragen der df gegen v aus der Direkten Methode
plt.clf()
plt.grid()
plt.xlim(-0.6, 0.6)
plt.xlabel(r"Geschwindigkeit $v\ [\mathrm{\frac{m}{s}}]$",
           fontsize=14, family='serif')
plt.ylim(-30, 50)
plt.ylabel(r"Frequenzdifferenz $\Delta \nu\ [\mathrm{Hz}]$",
           fontsize=14, family='serif')
plt.errorbar(noms(uv), noms(udf), yerr=stds(udf), fmt="rx", label="Messwerte")
plt.plot(V, y(V, *popt1), label="Regressionsgerade")
plt.plot(V, y1(V, *popt15), label="Regressionsgerade durch den Ursprung")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("Grafiken/Frequenzdifferenz_direkt.pdf")

### Ausftragen der Schwebungsmethode
plt.clf()
plt.grid()
plt.xlim(-0.6, 0.6)
plt.xlabel(r"Geschwindigkeit $v\ [\mathrm{\frac{m}{s}}]$",
           fontsize=14, family='serif')
plt.ylim(-32, 50)
plt.ylabel(r"Frequenzdifferenz $\Delta \nu\ [\mathrm{Hz}]$",
           fontsize=14, family='serif')
plt.errorbar(noms(uv_plot), noms(udF_plot), yerr=stds(udF_plot),
             fmt="rx", label="Messwerte")
plt.plot(V, y(V, *popt2), label="Regressionsgerade")
plt.plot(V, y1(V, *popt25), label="Regressionsgerade durch den Ursprung")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("Grafiken/Frequenzdifferenz_Schwebung.pdf")

# Student t-Test


def S(nx, ny, sx, sy):
    return np.sqrt(((((nx - 1) * sx**2 + (ny - 1) * sy**2) * (nx + ny)) /
                  ((nx + ny - 2) * (nx * ny))))


def T(x_avr, y_avr, s):
    return (x_avr - y_avr)/s

S1 = S(len(uk_inv), len(udf), stds(uk_inv_avr), stds(params1[0]))
T1 = T(noms(uk_inv_avr), noms(params1[0]), S1)

S2 = S(len(uk_inv), len(udF), stds(uk_inv_avr), stds(params2[0]))
T2 = T(noms(uk_inv_avr), noms(params2[0]), S2)

S3 = S(len(udf), len(udF), stds(params1[0]), stds(params2[0]))
T3 = T(noms(params1[0]), noms(params2[0]), S3)

## Print Funktionen

#File = open("Daten/Tabelle_Fahrtzeiten.tex", "w")
#
#File.write(lxtabs.toTable([G, ut_h1, ut_h2, ut_r1, ut_r2],
#                          col_titles=["Gang", "Zeit", "Zeit", "Zeit", "Zeit"],
#                          col_syms=[r"g", r"t_{h1}", r"t_{h2}",
#                                    r"t_{r1}", r"t_{r2}"],
#                          col_units=[r"", r"\second", r"\second",
#                                     r"\second", r"\second"],
#                          fmt=["c", "c", "c", "c", "c"],
#                          cap="Fahrtdauern des Wagens" +
#                              " in den verschiedenen Gängen",
#                          label="Auswertung_Fahrtzeiten"))
#File.close()
#
#File = open("Daten/Tabelle_Geschwindigkeiten.tex", "w")
#
#File.write(lxtabs.toTable([G, uv_h, uv_r],
#                          col_titles=["Gang", "Geschwindigkeit",
#                                      "Geschwindigkeit"],
#                          col_syms=[r"g", r"v_{h1}", r"v_{h2}"],
#                          col_units=[r"", r"\meter\per\second",
#                                     r"\meter\per\second"],
#                          fmt=["c", "c", "c"],
#                          cap="Geschwindigkeiten des Wagens" +
#                              " in den verschiedenen Gängen",
#                          label="Auswertung_Geschwindigkeiten"))
#File.close()

#File = open("Daten/Tabelle_Direkt.tex", "w")
#
#File.write(lxtabs.toTable([G, uf_h1, uf_h2, uf_r1, uf_r2],
#                          col_titles=["Gang", "Frequenz", "Frequenz",
#                                      "Frequenz", "Frequenz"],
#                          col_syms=[r"g", r"\nu_{h1}", r"\nu_{h2}",
#                                    r"\nu_{r1}", r"\nu_{r2}"],
#                          col_units=[r"", r"\hertz", r"\hertz",
#                                     r"\hertz", r"\hertz"],
#                          fmt=["c", "c", "c", "c", "c"],
#                          cap="Direkt gemessene Frequenzen des Wagens" +
#                              " in den verschiedenen Gängen",
#                          label="Auswertung_Frequenz_Direkt"))
#File.close()


#File = open("Daten/Tabelle_DirektDifferenzen.tex", "w")
#
#File.write(lxtabs.toTable([G, udf_h, udf_r],
#                          col_titles=["Gang", "Differenzfrequenz",
#                                      "Differenzfrequenz"],
#                          col_syms=[r"g", r"\Delta \nu_{h}",
#                                    r"\Delta \nu_{r}"],
#                          col_units=[r"", r"\hertz", r"\hertz"],
#                          fmt=["c", "c", "c"],
#                          cap="Frequenzänderungen des Wagens nach der" +
#                              "Direkten Methode in den verschiedenen Gängen",
#                          label="Auswertung_Frequenzänderung_Direkt"))
#File.close()


#File = open("Daten/Tabelle_Schwebung.tex", "w")
#
#File.write(lxtabs.toTable([G, udf_h1, udf_h2,udF_h_avr,
#                           udf_r1, udf_r2, udF_r_avr],
#                          col_titles=["Gang", "Frequenzdifferenz",
#                                      "Frequenzdifferenz",
#                                       "Frequenzdifferenz",
#                                      "Frequenzdifferenz",
#                                      "Frequenzdifferenz",
#                                      "Frequenzdifferenz"],
#                          col_syms=[r"g", r"\Delta\nu_{h1}",
#                                    r"\Delta\nu_{h2}",
#                                    r"\left<\Delta\nu_{h}\right>",
#                                    r"\Delta\nu_{r1}", r"\Delta\nu_{r2}",
#                                    r"\left<\Delta\nu_{r}\right>"],
#                          col_units=[r"", r"\hertz", r"\hertz",
#                                     r"\hertz", r"\hertz",
#                                     r"\hertz", r"\hertz"],
#                          fmt=["c", "c", "c", "c", "c", "c", "c"],
#                          cap="Direkt gemessene Frequenzen des Wagens" +
#                              " in den verschiedenen Gängen",
#                          label="Auswertung_Frequenz_Schwebung"))
#File.close()

#File = open("Daten/Tabelle_Ruhefrequenz.tex", "w")
#
#File.write(lxtabs.toTable([uf0],
#                          col_titles=["Ruhefrequenz"],
#                          col_syms=[r"\nu_{0}"],
#                          col_units=[r"\hertz"],
#                          fmt=["c"],
#                          cap="Gemessene Ruhefrequenzen",
#                          label="Auswertung_Ruhefrequenz"))
#File.close()

#File = open("Daten/Tabelle_Wellenlaenge.tex", "w")
#
#File.write(lxtabs.toTable([us, n, uk, uk_inv],
#                          col_titles=["Strecke",
#                                      "Wellenlängenanzahl", "Wellenlänge",
#                                      "inverse Wellenlänge"],
#                          col_syms=[r"s", "n", r"\lambda",
#                                    r"\lambda{-1}"],
#                          col_units=[r"\meter", r"",
#                                     r"\meter", r"\per\meter"],
#                          fmt=["c", "c", "c", "c"],
#                          cap="Messdaten der Wellenlängenbestimmung und " +
#                              "Wellenlänge sowie inverse Wellenlänge",
#                          label="Auswertung_Wellenlänge"))
#File.close()
