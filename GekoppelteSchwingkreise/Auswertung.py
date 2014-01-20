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

sys.path.append("..\_globales\python")
import latextables as lxtabs


def resFreq_plus(c, c_sp, l):
    return 1/(2 * const.pi * (l * (c + c_sp))**(0.5))


def resFreq_minus(c, c_k, c_sp, l):
    return 1/(2 * const.pi * (l * ((1/(1/c + 2/c_k)) + c_sp))**(0.5))





# Laden der Bauteildaten Kapazität, Spulenkapazität, Induktivität
C, C_sp, L = np.loadtxt("Messdaten/Bauteile.txt", unpack=True)

# Laden der Fehler
C_err, L_err = np.loadtxt("Messdaten/Bauteile _Fehler.txt",
                          unpack=True)

# Fehlerbehaftete Bauteile
uC = ufloat(C, C_err)
uL = ufloat(L, L_err)

# Skalierung
uC *= 1e-09  # [nF] > [F]
C_sp *= 1e-09  # [nF] > [F]
uL *= 1e-03  # [mH] > [H]


# Laden der gemessenen Resonanzfrequenz
f_0 = np.loadtxt("Messdaten/Resonanzfrequenz.txt")

# Laden des Fehlers der Frequenz
f_err = np.loadtxt("Messdaten/Resonanzfrequenz_Fehler.txt")

# Fehlerbehaftete Resonanzfrequenz
uf_0 = ufloat(f_0, f_err)

# Skalieren
uf_0 *= 1e03  # [kHz] > [Hz]

# Laden des Kapazitätfehlers
C_err = np.loadtxt("Messdaten/Kondensator_Fehler.txt")

# Skalierung
C_err *= 1e-02  # [%] > [1]

# Laden der Kapazität und der Anzahl der Amplituden pro Amplitude
C1, Amps = np.loadtxt("Messdaten/Schwebung.txt", unpack=True)

# Fehlerbehaftete Kapazitäten
uC1 = unp.uarray(C1, (len(C1) * [C_err]) * C1)

# Skalierung
uC1 *= 1e-09  # [nF] > [F]

# Bestimmen der Resonanzfrequenz f+
uF_p = resFreq_plus(uC, C_sp, uL)

# Bestimmung der Resonanzfrequenz f-
uF_m = resFreq_minus(uC, uC1, C_sp, uL)

# Bestimmung der gemessenen Verhältnisse
Amps_rel = 1/Amps

# Bestimmung der berechneten Verhältnisse
f_si = 0.5 * (uF_p + uF_m)  # Schwebungsfrequenz
f_sb = (uF_m - uF_p)  # Schwingungsfrequenz

f_rel = f_sb/f_si

# Relative Abweichung der Messwerte
Df_rel = (Amps_rel/f_rel - 1)
Df_rel_avr = np.mean(Df_rel)

# Laden der Kapazität und der Frequenzen f+, f-
C2, f_p, f_m = np.loadtxt("Messdaten/Resonanzen.txt", unpack=True)

# Fehlerbehaftete Kapazitäten
uC2 = unp.uarray(C2, (len(C2) * [C_err]) * C2)

# Fehler behaftete Frequenzen
uf_p = unp.uarray(f_p, len(f_p) * [f_err])
uf_m = unp.uarray(f_m, len(f_m) * [f_err])


# Skalierung
uC2 *= 1e-09
uf_p *= 1e03
uf_m *= 1e03

# Vehältnis der gemessenen und berechnetne Frequenz f+
uf_p_rel = uf_p / uF_p

# Verhältnis der gemessenen und berechneten Frequenz f-
uf_m_rel = uf_m / uF_m


# Laden der Wobbelgenerator Einstellungen
dx, dy, f_min, f_max, t = np.loadtxt("Messdaten/Wobbelgenerator" +
                                     "_Einstellungen.txt", unpack=True)

# Skalierung [kHz] > [Hz]
f_min *= 1e03
f_max *= 1e03

# Breite des Frequenzspektrums
df = f_max - f_min

# Kasten pro Spektrumsdurchlauf
Ticks_df = t / dx

# Frequenz pro tick
f_Ticks = df / Ticks_df


def Z(w, l, c, ck):
    return ((w * l) - (1/c + 1/ck)/w)


def Strom(u, w, c, ck, r, l):
    return (u/(4 * w**2 * ck**2 * r**2 * Z(w, l, c, ck)**2 +
           (1/(w*ck) - w * ck * Z(w, l, c, ck)**2 +
            w * r**2 * ck)**2)**(0.5))



# Laden der Wobbelgenerator Messung (aus den Grafiken bestimmt)
C3, X_p, Y_p, X_m, Y_m = np.loadtxt("Messdaten/Wobbelgenerator.txt",
                                    unpack=True)


# Skalierung
C3 *= 1e-09  # [F]


R = 78  # [Ohm]

# Fehlerbehaftete Kapazitäten
uC3 = unp.uarray(C3, (len(C3) * [C_err]) * C3)

# Fehlerbehaftete Maße
XY_err = 0.1
uX_p = unp.uarray(X_p, len(X_p)*[XY_err])
uY_p = unp.uarray(Y_p, len(Y_p)*[XY_err])
uX_m = unp.uarray(X_m, len(X_m)*[XY_err])
uY_m = unp.uarray(Y_m, len(Y_m)*[XY_err])


# Skalieren der X-Achse (Frequenzen)
uX_p *= f_Ticks
uX_p += f_min

uX_m *= f_Ticks
uX_m += f_min

# Skalierung der Y-Achse (Spannung)
uY_p *= dy
uY_m *= dy

Xrange = np.arange(f_min, 2 * f_max, 20)

Ip = Strom(12, uX_p * 2 * const.pi, uC, uC3, R, uL)
#Ip = Strom(12, uX_p * const.pi, uC, uC3, R, uL)
Im = Strom(12, uX_m * 2 * const.pi, uC, uC3, R, uL)
Ip_calc = uY_p/R
Im_calc = uY_m/R

n = 1
for c in uC3:
    plt.clf()
    plt.xlabel("Frequenz $f\,[\mathrm{kHz}]$")
    plt.xlim(2e04, 6e04)
    plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter
                                       (lambda x, _: float(x * 1e-03)))
    plt.ylabel("Stromstärke $I\,[\mathrm{A}]$")
    plt.plot(Xrange, noms(Strom(12, Xrange * 2 * const.pi, uC, c, R, uL)))
    plt.savefig("Grafiken/Stromverlauf{}.pdf".format(str(n)))
    n += 1

plt.clf()
plt.xlabel("Frequenz $f\,[\mathrm{kHz}]$")
plt.xlim(2e04, 6e04)
plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter
                                    (lambda x, _: float(x * 1e-03)))
plt.ylabel("Stromstärke $I\,[\mathrm{A}]$")
plt.plot(Xrange, noms(Strom(12, Xrange * 2 * const.pi, uC, uC3[0], R, uL)))



## Print Funktionen


#f = open("Daten/Tabelle_Schwebung.tex", "w")
#
#f.write(lxtabs.toTable([uC1*1e09, Amps, Amps_rel],
#        col_titles=["Kapazität", "Amplitudenzahl", "Amplitudenverhältnis"],
#        col_syms=["C", "A_{1}", r"\frac{A_{0}}{A_{1}}"],
#        col_units=["nF", "", ""],
#        fmt=["c", "c", "c"],
#        cap="Messwerte zur Bestimmung des Frequenzverhältnisses",
#        label="Schwebung"))
#
#f.close()

#f = open("Daten/Tabelle_FrequenzVerhältnis.tex", "w")
#
#f.write(lxtabs.toTable([Df_rel],
#        col_titles=["Relativeabweichung"],
#        col_syms=[r"\frac{\envert{\nu_{rel, mess} - \nu_{rel, theo}}}" +
#                  r"{\nu_{rel, theo}}"],
#        col_units=[""],
#        fmt=["c"],
#        cap="Relative Abweichung der gemessenen Frequenzverhältnisse",
#        label="Schwebung"))
#
#f.close()

#f = open("Daten/Tabelle_Fundamental.tex", "w")
#
#_aux = [uF_p] * len(uF_m)
#_aux = np.array(_aux)
#f.write(lxtabs.toTable([uC2 * 1e09, _aux * 1e-03, uF_m * 1e-03, f_rel],
#        col_titles=["Kapazitäten", "Fundamentalfrequenz",
#                    "Fundamentalfrequenz", "Frequenzverhältnis"],
#        col_syms=["C_{K}", r"\nu^{+}", r"\nu^{-}",
#                  r"\tfrac{2(\nu^{-} - \nu^{+})}{\nu^{-} + \nu^{+}}"],
#        col_units=[r"\nano\farad", r"\kilo\hertz", r"\kilo\hertz", ""],
#        fmt=["c", "c", "c", "c"],
#        cap="Fundamentalfrequenzen und das Frequenzverhältnis der Schwebung",
#        label="Fundamental_Freqs"))
#
#f.close()

#f = open("Daten/Tabelle_FundamentalMessung.tex", "w")
#
#_aux = [uF_p] * len(uF_m)
#_aux = np.array(_aux)
#f.write(lxtabs.toTable([_aux * 1e-03, uF_m * 1e-03,
#                        uf_p * 1e-03, uf_m * 1e-03,
#                        uf_p_rel, uf_m_rel],
#        col_titles=["Fundamentalfrequenz",
#                    "Fundamentalfrequenz",
#                    "Fundamentalfrequenz",
#                    "Fundamentalfrequenz",
#                    "Frequenzverhältnis",
#                    "Frequenzverhältnis"],
#        col_syms=[r"\nu^{+}", r"\nu^{-}", r"\nu^{+}", r"\nu^{-}",
#                  r"\tfrac{\nu^{+}}{\nu^{+}_{theo}}",
#                  r"\tfrac{\nu^{-}}{\nu^{-}_{theo}}"],
#        col_units=[r"\kilo\hertz", r"\kilo\hertz", r"\kilo\hertz",
#                   r"\kilo\hertz", "", ""],
#        fmt=["c", "c", "c", "c", "c", "c"],
#        cap="Berechnete und gemessene Fundamentalfrequenzen mit jeweiligem" +
#            "Verhältnis",
#        label="Fundamental_Messung"))
#
#f.close()

#f = open("Daten/Tabelle_WobbelVerlauf.tex", "w")
#f.write(lxtabs.toTable([uC3 * 1e09, uX_p * 1e-03, uX_m * 1e-03,
#                        uY_p, uY_m],
#        col_titles=["Kapazitäten", "Fundamentalfrequenz",
#                    "Fundamentalfrequenz", "Spannung", "Spannung"],
#        col_syms=["C_{K}", r"\nu^{+}", r"\nu^{-}",
#                  "U^{+}", "U^{-}"],
#        col_units=[r"\nano\farad", r"\kilo\hertz", r"\kilo\hertz",
#                   r"\volt", r"\volt"],
#        fmt=["c", "c", "c", "c", "c"],
#        cap="Fundamentalfrequenzen und jeweilige Spannungsspitzen",
#        label="WobbelVerlauf"))
#f.close()

#f = open("Daten/Tabelle_Strom2.tex", "w")
#f.write(lxtabs.toTable([Ip, Im, Ip_calc, Im_calc],
#        col_titles=["Stromstärke", "Stromstärke",
#                    "Stromstärke", "Stromstärke"],
#        col_syms=["I^{+}", "I^{-}", "I^{+}", "I^{-}"],
#        col_units=[r"\ampere", r"\ampere", r"\ampere", r"\ampere"],
#        fmt=["c", "c", "c", "c"],
#        cap="Theoretisch bestimmte und gemessene Stromstärken",
#        label="I2"))
#f.close()