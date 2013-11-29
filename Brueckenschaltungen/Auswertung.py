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

### Uncertianties Funktionen
umean = unc.wrap(np.mean)


# Steuermakros
PRINT = True

# Maximalwiderstand des Potentiometers
R_max = np.loadtxt("Messdaten/Potentiometer.txt")


# Funktion zur Berechnung von R4 aus R3
def R4(r):
    return (R_max - r)

### Wheatstonebrücke (1)

## Abgleichwiderstände R2 und Potentiometerwiderstand R3
R2_1, R3_1 = np.loadtxt("Messdaten/Wheatstone.txt", unpack=True)

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
uRx_1 = R2_1 * uR34_1

# Mittlewert für Rx
uRx_1_avr = umean(uRx_1)


#print(std(noms(uRx_1))/len(uRx_1))

### Kapazitätmessbrücke ideal (2)

## Abgleichkapazitäten und Potntiometerwiderstände
C2_2, R3_2 = np.loadtxt("Messdaten/Kapazitaetsmessung_ideal.txt", unpack=True)

## Berechnung von R4 aus R3
R4_2 = R4(R3_2)

## Berechnung des Quotienten R3/R4
R34_2 = R3_2/R4_2

# Fehlerbehafteter Quotient
uR34_2 = unp.uarray(R34_2, R34_2 * R34_err)

## Berechnug der Unbekannten Kapazität Cx
uCx_2 = C2_2 / (uR34_2)

# Mittelwert für Cx
uCx_2_avr = umean(uCx_2)

### Kapatitätsmessbrücke real (3)

## Abgleichkapazität, Potentiometerwiderstand
C2_3, R3_3 = np.loadtxt("Messdaten/Kapazitaetsmessung_real.txt",
                        usecols=(0, 1), unpack=True)

## Fehler des Stellgliedes R2
R2_err = np.loadtxt("Messdaten/Messfehler.txt", usecols=(1, 1))[0]  # [%]

R2_err *= 1e-02  # [1]

## Stellglied R2
R2_3 = np.loadtxt("Messdaten/Kapazitaetsmessung_real_R2.txt")

# Fehlerbehaftetes Stellglied
uR2_3 = ufloat(R2_3, R2_3*R2_err)

## Bestimmung der R4 aus den R3
R4_3 = R4(R3_3)

## Bildung des Quotienten R3/R4
R34_3 = R3_3 / R4_3


# Fehlerbehafteter Quotient
uR34_3 = unp.uarray(R34_3, R34_3 * R34_err)

## Bestimmung der unbekannten Kapazität
uCx_3 = C2_3 / uR34_3

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
uLx_4 = L2_4 * uR34_4

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
uR3_5 = ufloat(R3_5, R3_5_err)

uR2_5 = ufloat(R2_5, R2_5*R2_err)

## Berechnen der R4 aus den R3
R4_5 = R4(R3_5)

# Fehlerbehafteter Widerstand
uR4_5 = ufloat(R4_5, R4_5 * R4_5_err)


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


## Plot von U/Uq gegen f/f0
def TKurve(O):
    return np.sqrt((O**2 - 1)**2 / (9*(1 - O**2)**2 + 81 * O**2))

x = np.linspace(1e-03, 1e03,
                num = 1000000)


plt.clf()
plt.grid()
plt.xlabel(r"$\Omega = \frac{\nu}{\nu_{0}}$", fontsize=18)
plt.ylabel(r"$\frac{U_{Br}}{U_{S}}$", fontsize=20)
plt.xscale("log")
plt.xlim(1e-02,1e02)
plt.ylim(0, 0.4)

plt.plot(noms(uf)/noms(uf_min), noms(uU)/noms(uUq), "rx", label="Messwerte" )
plt.plot(x, TKurve(x), color="grey",
         label="Theoriekurve")

plt.legend(loc="lower left")
plt.tight_layout()

plt.savefig("Grafiken/WienRobinson.pdf")


### Berechnung des Klirrfaktors

## Bestimmung der Oberwellenamplitude
uU2 = min(uU)/TKurve(2)

## Bestimmung des Klirrfaktors
uk = uU2/min(uU)

print(uU2)
print(uk)

## Print Funktionen
PRINT = False
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
    print("\n-Berechne Kapazität:\n", uLx_5)
    print("\n-Berechner Widerstand:\n", uRx_5)

    print("\n Theoretische Minimalspanungsfrequenz:\n", f_0)