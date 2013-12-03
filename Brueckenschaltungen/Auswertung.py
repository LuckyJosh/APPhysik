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
# Eigene Funktionen
sys.path.append("..\globales\python")
from latextables import toTable as tab

### Uncertianties Funktionen
umean = unc.wrap(np.mean)
usqrt = unc.wrap(np.sqrt)

# Steuermakros
PRINT = True
TABS = True

# Maximalwiderstand des Potentiometers
R_max = np.loadtxt("Messdaten/Potentiometer.txt")
X2_err = np.loadtxt("Messdaten/Messfehler.txt", usecols=(5, 5))[0]  # [%]
X2_err *= 1e-02  # [1] 

# Funktion zur Berechnung von R4 aus R3
def R4(r):
    return (R_max - r)


uR4 = unc.wrap(R4)
### Wheatstonebrücke (1)

## Abgleichwiderstände R2 und Potentiometerwiderstand R3
R2_1, R3_1 = np.loadtxt("Messdaten/Wheatstone.txt", unpack=True)

#Fehlerbehaftetes  R2
uR2_1 = unp.uarray(R2_1, R2_1 * X2_err)

## Fehler des Quotienten R3/R4
R34_err = np.loadtxt("Messdaten/Messfehler.txt", usecols=(0, 0))[0]  # [%]
R34_err *= 1e-02  # [1]

## Berechnung von R4 aus R3
R4_1 = R4(R3_1)

## Berechnung des Quotienten R3/R4
R34_1 = R3_1/R4_1

#Fehlerbehafteter Quotient
uR34_1 = unp.uarray(R34_1, R34_1 * R34_err)

## Berechnung des Unbekannten Widerstands
uRx_1 = uR2_1 * uR34_1

# Mittlewert für Rx
uRx_1_avr = umean(uRx_1)


#print(std(noms(uRx_1))/len(uRx_1))

### Kapazitätmessbrücke ideal (2)

## Abgleichkapazitäten und Potntiometerwiderstände
C2_2, R3_2 = np.loadtxt("Messdaten/Kapazitaetsmessung_ideal.txt", unpack=True)

#Fehlerbehaftetes  C2
uC2_2 = unp.uarray(C2_2, C2_2 * X2_err)

## Berechnung von R4 aus R3
R4_2 = R4(R3_2)

## Berechnung des Quotienten R3/R4
R34_2 = R3_2/R4_2

# Fehlerbehafteter Quotient
uR34_2 = unp.uarray(R34_2, R34_2 * R34_err)

## Berechnug der Unbekannten Kapazität Cx
uCx_2 = uC2_2 / (uR34_2)

# Mittelwert für Cx
uCx_2_avr = umean(uCx_2)

### Kapatitätsmessbrücke real (3)

## Abgleichkapazität, Potentiometerwiderstand
C2_3, R3_3 = np.loadtxt("Messdaten/Kapazitaetsmessung_real.txt",
                        usecols=(0, 1), unpack=True)
#Fehlerbehaftetes  C2
uC2_3 = unp.uarray(C2_3, C2_3 * X2_err)

## Fehler des Stellgliedes R2
R2_err = np.loadtxt("Messdaten/Messfehler.txt", usecols=(1, 1))[0]  # [%]

R2_err *= 1e-02  # [1]

## Stellglied R2
R2_3 = np.loadtxt("Messdaten/Kapazitaetsmessung_real_R2.txt")

# Fehlerbehaftetes Stellglied
uR2_3 = ufloat(R2_3, R2_3*R2_err)
print(uR2_3)

## Bestimmung der R4 aus den R3
R4_3 = R4(R3_3)

## Bildung des Quotienten R3/R4
R34_3 = R3_3 / R4_3


# Fehlerbehafteter Quotient
uR34_3 = unp.uarray(R34_3, R34_3 * R34_err)

## Bestimmung der unbekannten Kapazität
uCx_3 = uC2_3 / uR34_3

# Bestimmung des Mittelwertes von uCx_3
uCx_3_avr = umean(uCx_3)

## Bestimmung des Unbekannten Widerstands der Kapazität
uRx_3 = R2_3 * uR34_3

# Bestimmung des Mittelwertes von uRx_3
uRx_3_avr = umean(uRx_3)


### Induktivitätsmessbrücke (4)

## Abgleichinduktivitäten, Potentiometer Widerstand
L2_4, R3_4 = np.loadtxt("Messdaten/Induktivitaet.txt", usecols=(0, 1),
                        unpack=True)

#Fehlerbehaftetes  L2
uL2_4 = unp.uarray(L2_4, L2_4 * X2_err)

## Stellglied R2
R2_4 = np.loadtxt("Messdaten/Induktivitaet_R2.txt")

# Fehlerbehaftetes Stellglied
uR2_4 = ufloat(R2_4, R2_4*R2_err)

## Berechnung der R4 aus den R3
R4_4 = R4(R3_4)

## Berechnung des Quotienten R3/R4
R34_4 = R3_4/R4_4

# Fehlerbehafteter Quotient
uR34_4 = unp.uarray(R34_4, R34_4 * R34_err)

## Berechnung der unbekannten Induktivität Lx
uLx_4 = uL2_4 * uR34_4

# Berechnung  des Mittelwertes
uLx_4_avr = umean(uLx_4)

## Berechnung des unbekannten Widerstands
uRx_4 = uR2_4 * uR34_4

# Berechnung des Mittelwerts
uRx_4_avr = umean(uRx_4)


### Maxwellbrücke

## Fehler von R3 und R4

R3_5_err, R4_5_err = np.loadtxt("Messdaten/Messfehler.txt", usecols=(2, 2),
                                unpack=True)

R3_5_err *= 1e-02  # [1]
R4_5_err *= 1e-02  # [1]

## Stellglied (R), Stellglied (C), Potentiometerwiderstand
R2_5, C4_5, R3_5 = np.loadtxt("Messdaten/Maxwell.txt")

# Fehlerbehafteter Widerstand
uR3_5 = ufloat(R3_5, R3_5*R3_5_err)

uR2_5 = ufloat(R2_5, R2_5*R2_err)

## Berechnen der R4 aus den R3
uR4_5 = uR4(uR3_5)

# Fehlerbehafteter Widerstand
uR4_5 = ufloat(noms(uR4_5), stds(uR4_5) + noms(uR4_5) * R4_5_err)


## Berechnung der unbekannten Induktivität
uLx_5 = uR2_5 * uR3_5 * C4_5


## Berechnung des unbekannten Widerstands
uRx_5 = uR2_5 * uR3_5 / uR4_5


### Wien-Roninson Brücke (6)


## Frequenzen und Spannungen
f, U = np.loadtxt("Messdaten/WienRobinson.txt", unpack=True, usecols=(0, 1))

## Quellspannung
Uq = np.loadtxt("Messdaten/WienRobinson_Uq.txt")

## Fehler der Frequenzen
f_err, U_err = np.loadtxt("Messdaten/Messfehler.txt", usecols=(3, 4))

## Fehlerbehaftete Größen
uf = unp.uarray(f, f_err)
uU = unp.uarray(U, U_err)
uUq = ufloat(Uq, U_err)

## Bestimmen der Frequenz mit minimaler Spannung
uf_min = uf[(np.where(min(uU) == uU)[0])[0]]


#TODO: 332 Ohm laden
## Bestimmung der theoretischen "minimal Frequenz"
f_0 = 1/(2 * const.pi * 332 * uCx_2_avr)*1e09


## Abweichung von der Theorie

df = abs(noms(uf_min) - noms(f_0))/noms(f_0)

## Plot von U/Uq gegen f/f0
def TKurve(O):
    return usqrt((O**2 - 1)**2 / (9*((1 - O**2)**2 + 9 * O**2)))

x = np.linspace(1e-03, 1e03,
                num = 1000000)


plt.clf()
plt.grid()
plt.xlabel(r"$\Omega = \frac{\nu}{\nu_{0}}$", fontsize=18)
plt.ylabel(r"$\frac{U_{Br}}{U_{S}}$", fontsize=20)
plt.xscale("log")
plt.xlim(1e-02,1e02)
plt.ylim(0, 0.4)

X = uf/uf_min
Y = uU/uUq

uX = unp.uarray(noms(X), stds(X))
uY = unp.uarray(noms(Y), stds(Y))

plt.errorbar(noms(uX), noms(uY),
             xerr=stds(uX), yerr=stds(uY), 
             fmt="rx", label="Messwerte" )
plt.plot(x, TKurve(x), color="grey",
         label="Theoriekurve")

plt.legend(loc="lower left")
plt.tight_layout()

plt.savefig("Grafiken/WienRobinson.pdf")


### Berechnung des Klirrfaktors

## Bestimmung der Oberwellenamplitude
uU2 = min(uU)/TKurve(2)
uU2 = ufloat(noms(uU2), stds(uU2))
print(TKurve(2))
## Bestimmung des Klirrfaktors
uk = uU2/uUq

print(uU2)
print(min(uU))
print(uk)



## Print Funktionen
#PRINT = False
if PRINT:
    print("\nWheatstone:")
    print("\n-Widerstandsquotient:\n", uR34_1)
    print("\n-Berechneter Widerstand:\n", uRx_1)
    print("\t-Mittelwert:", uRx_1_avr)

    print("\nKapazitätsmessung ideal:")
    print("\n-Widerstandsquotient:\n", uR34_2)
    print("\n-Berechne Kapazität:\n", uCx_2)
    print("\t-Mittelwert:", uCx_2_avr)

    print("\nKapazitätsmessung real:")
    print("\n-Widerstandsquotient:\n", uR34_3)
    print("\n-Berechne Kapazität:\n", uCx_3)
    print("\t-Mittelwert:", uCx_3_avr)
    print("\n-Berechner Widerstand:\n", uRx_3)
    print("\t-Mittelwert:", uRx_3_avr)

    print("\nInduktivitätsmessung real:")
    print("\n-Widerstandsquotient:\n", uR34_4)
    print("\n-Berechne Kapazität:\n", uLx_4)
    print("\t-Mittelwert:", uLx_4_avr)
    print("\n-Berechner Widerstand:\n", uRx_4)
    print("\t-Mittelwert:", uRx_4_avr)

    print("\nInduktivitätsmessung real:")
    print("\n-Berechne Kapazität:\n", uLx_5*1e-09)
    print("\n-Berechner Widerstand:\n", uRx_5)

    print("\nTheoretische Minimalspanungsfrequenz:\n", f_0,
          "\nAbweichung von der Theorie:\n", df)



if TABS:
    f1 = open("Daten/Tabelle_Wheatstone.tex", "w")
    f1.write(tab([uR2_1, R3_1, uR34_1, uRx_1],
                 ["Widerstand", "Widerstan ", "Quotient", " Widersta"],
                 ["R_{2}", "R_{3}", r"\frac{R_{3}}{R_{4}}", "R_{x}"],
                 [r"\ohm", r"\ohm", "", r"\ohm"],
                 ["c", "c", "c", "c"],
                 cap="Werte der Messung an der Wheatstonebrücke",
                 label="Wheatstone"))
    f1.close()

    f2 = open("Daten/Tabelle_Kapazitaet_ideal.tex", "w")
    f2.write(tab([uC2_2, R3_2, uR34_2, uCx_2],
                 ["Kapazität", "Widerstand", "Quotient", "Widers"],
                 ["C_{2}", "R_{3}", r"\frac{R_{3}}{R_{4}}", "C_{x}"],
                 [r"\nano\farad", r"\ohm", "", r"\nano\farad"],
                 ["c", "c", "c", "c"],
                 cap="Werte der Messung einer idealen Kapazität" +
                 "an der Kapazitätsmessbrücke",
                 label="Kapazitaet_ideal"))
    f2.close()

    f3 = open("Daten/Tabelle_Kapazitaet_real.tex", "w")
    f3.write(tab([uC2_3, R3_3, uR34_3, uCx_3, uRx_3],
                 ["Kapazität", "Widerstand", "Quotient", "Kapazitä", "Widersta"],
                 ["C_{2}", "R_{3}", r"\frac{R_{3}}{R_{4}}", "C_{x}", "R_{x}"],
                 [r"\nano\farad", r"\ohm", "", r"\nano\farad", r"\ohm"],
                 ["c", "c", "c", "c", "c"],
                 cap="Werte der Messung einer idealen Kapazität" +
                 "an der Kapazitätsmessbrücke",
                 label="Kapazitaet_real"))
    f3.close()
    f4 = open("Daten/Tabelle_Induktivitaet_Bruecke.tex", "w")
    f4.write(tab([uL2_4, R3_4, uR34_4, uLx_4, uRx_4],
                 ["Induktivität", "Widerstand", "Quotient", "Induktivitä", "Widersta"],
                 ["L_{2}", "R_{3}", r"\frac{R_{3}}{R_{4}}", "L_{x}", "R_{x}"],
                 [r"\milli\henry", r"\ohm", "", r"\milli\henry", r"\ohm"],
                 ["c", "c", "c", "c", "c"],
                 cap="Werte der Messung einer realen Induktivität" +
                 "mit einer Induktivitätsmessbrücke",
                 label="Induktivitaets_Bruecke"))
    f4.close()

    f5 = open("Daten/Tabelle_Frequenz.tex", "w")
    f5.write(tab([uf, uU],
                 ["Frequenz", "Brückenspannung"],
                 [r"\nu", "U_{Br}"],
                 [r"\hertz", r"\volt"],
                 ["c", "c"],
                 cap="Generatorfrequenzen und gemessene Brückenspannungen",
                 label="Frequenz"))
    f5.close()