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

sys.path.append("..\globales\python")
import latextables as lxtabs


# Laden der Bauteildaten Kapazität, Spulenkapazität, Induktivität
C, C_sp, L = np.loadtxt("Messdaten/Bauteile.txt", unpack=True)

# Laden der Fehler
C_err, L_err = np.loadtxt("Messdaten/Bauteile _Fehler.txt", unpack=True)

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


# Laden der Kapazität und der Anzahl der Amplituden pro Amplitude
C1, Amps = np.loadtxt("Messdaten/Schwebung.txt", unpack=True)


# Laden der Kapazität und der Frequenzen f+, f-
C2, f_p, f_m = np.loadtxt("Messdaten/Resonanzen.txt", unpack=True)

# Fehler behaftete Frequenzen
uf_p = unp.uarray(f_p, len(f_p) * [f_err])
uf_m = unp.uarray(f_m, len(f_m) * [f_err])


# Laden der Wobbelgenerator Einstellungen
dx, dy, f_min, f_max, t = np.loadtxt("Messdaten/Wobbelgenerator" +
                                     "_Einstellungen.txt", unpack=True)


# Laden der Wobbelgenerator Messung (aus den Grafiken bestimmt)
C3, X_p, Y_p, X_m, Y_m = np.loadtxt("Messdaten/Wobbelgenerator.txt",
                                    unpack=True)

# Fehlerbehaftete Maße
XY_err = 0.1
uX_p = unp.uarray(X_p, len(X_p)*[XY_err])
uY_p = unp.uarray(Y_p, len(Y_p)*[XY_err])
uX_m = unp.uarray(X_m, len(X_m)*[XY_err])
uY_m = unp.uarray(Y_m, len(Y_m)*[XY_err])





## Print Funktionen 