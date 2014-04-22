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
#mpl.use("Qt4Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit
import sympy as sp
import sys
import uncertainties as unc
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)


from aputils.utils import Quantity, ErrorEquation, OutputFile
from aputils.latextables.tables import Table

# Erzeugen eines logs des stdout
sys.stdout = OutputFile("Daten/log.txt")

PRINT = True

#==============================================================================
#
# Elektronen im Elektrischen Feld
#
#==============================================================================

# Import der Ablenkspannungen
U_D = np.array(np.loadtxt("Messdaten/Ablenkspannung.txt", unpack=True))

# Import des Fehlers
u_d_err = np.loadtxt("Messdaten/Fehler_Ablenkspannung.txt")

# Erzeugung der einzelen Messreihen, löschen der unbesetzten Stellen
U_D_1 = U_D[0]
U_D_2 = U_D[1]
U_D_3 = U_D[2]
U_D_4 = np.delete(U_D[3], (np.where(U_D[3] == 0))[0])
U_D_5 = np.delete(U_D[4], (np.where(U_D[4] == 0))[0])


# Erstellen der fehlerbehafteten Größen
U_D_1_err = unp.uarray(U_D_1, len(U_D_1)*[u_d_err])
U_D_2_err = unp.uarray(U_D_2, len(U_D_2)*[u_d_err])
U_D_3_err = unp.uarray(U_D_3, len(U_D_3)*[u_d_err])
U_D_4_err = unp.uarray(U_D_4, len(U_D_4)*[u_d_err])
U_D_5_err = unp.uarray(U_D_5, len(U_D_5)*[u_d_err])


# Import der Beschleunigungsspannungen
U_B = np.loadtxt("Messdaten/Beschleunigungsspannung_E.txt", unpack=True)

# Import des Fehlers
u_b_err = np.loadtxt("Messdaten/Fehler_Beschleunigungsspannung.txt")

# Erzeugen der fehlerbehafteten Größen
U_B_err = unp.uarray(U_B, len(U_B)*[u_b_err])

# Berechnung der Distanzen
D = np.arange(0, 9, 1) * (const.inch/4) * 1e02  # cm

#Fehler der Distanzen
d_err = 0.2 * (const.inch/4) * 1e02  # cm

# Berechnung der fehlerbehafteten Größen
D_err = unp.uarray(D, [d_err]*len(D))

# Erstellen des Plots zur Bestimmung der Empfindlichkeit

# Definition einer Geraden
def gerade(x, m, b):
    return m * x + b


#==============================================================================
# Messreihe I
#==============================================================================

## Berechnung der Ausgleichsgerade
popt_1, pcov_1 = curve_fit(gerade, noms(U_D_1_err),
                           noms(D_err), sigma=stds(D_err))
error_1 = np.sqrt(np.diag(pcov_1))
param_m_1 = ufloat(popt_1[0], error_1[0])
param_b_1 = ufloat(popt_1[1], error_1[1])

print("Ausgleichsgerade I: m, b:", param_m_1, param_b_1) if PRINT else print()


## Messdaten
plt.errorbar(noms(U_D_1_err), noms(D_err),
             xerr=stds(U_D_1_err), yerr=stds(D_err),
             fmt="rx", label="Messwerte")

## Ausgleichsgrade
X = np.linspace(-40, 40, 200)
plt.plot(X, gerade(X, *popt_1), color="gray", label="Regressionsgeraden")

## Ploteinstellungen
plt.grid()
plt.xlim(-40, 25)
plt.ylim(-1, 6)
plt.xlabel(r"Ablenkspannung $U_d \ [\mathrm{V}]$", fontsize=14, family='serif')
plt.ylabel(r"Verschiebung $D \ [\mathrm{cm}]$", fontsize=14, family='serif')
plt.legend(loc="best")
plt.tight_layout()
#plt.show()
plt.savefig("Grafiken/EFeld_Messreihe_I.pdf")
plt.clf()

#==============================================================================
# Messreihe II
#==============================================================================

## Berechnung der Ausgleichsgerade
popt_2, pcov_2 = curve_fit(gerade, noms(U_D_2_err),
                           noms(D_err), sigma=stds(D_err))
error_2 = np.sqrt(np.diag(pcov_2))
param_m_2 = ufloat(popt_2[0], error_2[0])
param_b_2 = ufloat(popt_2[1], error_2[1])

print("Ausgleichsgerade II: m, b:", param_m_2, param_b_2) if PRINT else print()


## Messdaten
plt.errorbar(noms(U_D_2_err), noms(D_err),
             xerr=stds(U_D_2_err), yerr=stds(D_err),
             fmt="rx", label="Messwerte")

## Ausgleichsgrade
X = np.linspace(-40, 40, 200)
plt.plot(X, gerade(X, *popt_2), color="gray", label="Regressionsgerade")

## Ploteinstellungen
plt.grid()
plt.xlim(-40, 25)
plt.ylim(-1, 6)
plt.xlabel(r"Ablenkspannung $U_d \ [\mathrm{V}]$", fontsize=14, family='serif')
plt.ylabel(r"Verschiebung $D \ [\mathrm{cm}]$", fontsize=14, family='serif')
plt.legend(loc="best")
plt.tight_layout()
#plt.show()
plt.savefig("Grafiken/EFeld_Messreihe_II.pdf")
plt.clf()

#=============================================================================
#  Messreihe III
#==============================================================================

## Berechnung der Ausgleichsgerade
popt_3, pcov_3 = curve_fit(gerade, noms(U_D_3_err),
                           noms(D_err), sigma=stds(D_err))
error_3 = np.sqrt(np.diag(pcov_3))
param_m_3 = ufloat(popt_3[0], error_3[0])
param_b_3 = ufloat(popt_3[1], error_3[1])

print("Ausgleichsgerade III: m, b:", param_m_3, param_b_3) if PRINT else print()


## Messdaten
plt.errorbar(noms(U_D_3_err), noms(D_err),
             xerr=stds(U_D_3_err), yerr=stds(D_err),
             fmt="rx", label="Messreihe")

## Ausgleichsgrade
X = np.linspace(-40, 40, 200)
plt.plot(X, gerade(X, *popt_3), color="gray", label="Regressionsgerade")

## Ploteinstellungen
plt.grid()
plt.xlim(-40, 25)
plt.ylim(-1, 6)
plt.xlabel(r"Ablenkspannung $U_d \ [\mathrm{V}]$", fontsize=14, family='serif')
plt.ylabel(r"Verschiebung $D \ [\mathrm{cm}]$", fontsize=14, family='serif')
plt.legend(loc="best")
plt.tight_layout()
#plt.show()
plt.savefig("Grafiken/EFeld_Messreihe_III.pdf")
plt.clf()

#==============================================================================
#  Messreihe IV
#==============================================================================

## Berechnung der Ausgleichsgerade
popt_4, pcov_4 = curve_fit(gerade, noms(U_D_4_err),
                           noms(D_err[1:]), sigma=stds(D_err[1:]))
error_4 = np.sqrt(np.diag(pcov_4))
param_m_4 = ufloat(popt_4[0], error_4[0])
param_b_4 = ufloat(popt_4[1], error_4[1])

print("Ausgleichsgerade IV: m, b:", param_m_4, param_b_4) if PRINT else print()


## Messdaten
plt.errorbar(noms(U_D_4_err), noms(D_err[1:]),
             xerr=stds(U_D_4_err), yerr=stds(D_err[1:]),
             fmt="rx", label="Messwerte")

## Ausgleichsgrade
X = np.linspace(-40, 40, 200)
plt.plot(X, gerade(X, *popt_4), color="gray", label="Regressionsgerade")


## Ploteinstellungen
plt.grid()
plt.xlim(-40, 25)
plt.ylim(-1, 6)
plt.xlabel(r"Ablenkspannung $U_d \ [\mathrm{V}]$", fontsize=14, family='serif')
plt.ylabel(r"Verschiebung $D \ [\mathrm{cm}]$", fontsize=14, family='serif')
plt.legend(loc="best")
plt.tight_layout()
#plt.show()
plt.savefig("Grafiken/EFeld_Messreihe_IV.pdf")
plt.clf()

#==============================================================================
#  Messreihe V
#==============================================================================

## Berechnung der Ausgleichsgerade
popt_5, pcov_5 = curve_fit(gerade, noms(U_D_5_err),
                           noms(D_err[2:]), sigma=stds(D_err[2:]))
error_5 = np.sqrt(np.diag(pcov_5))
param_m_5 = ufloat(popt_5[0], error_5[0])
param_b_5 = ufloat(popt_5[1], error_5[1])

print("Ausgleichsgerade V: m, b:", param_m_5, param_b_5) if PRINT else print()


## Messdaten
plt.errorbar(noms(U_D_5_err), noms(D_err[2:]),
             xerr=stds(U_D_5_err), yerr=stds(D_err[2:]),
             fmt="rx", label="Messwerte")

## Ausgleichsgrade
X = np.linspace(-40, 40, 200)
plt.plot(X, gerade(X, *popt_5), color="gray", label="Regressionsgerade")

## Ploteinstellungen
plt.grid()
plt.xlim(-40, 25)
plt.ylim(-1, 6)
plt.xlabel(r"Ablenkspannung $U_d \ [\mathrm{V}]$", fontsize=14, family='serif')
plt.ylabel(r"Verschiebung $D \ [\mathrm{cm}]$", fontsize=14, family='serif')
plt.legend(loc="best")
plt.tight_layout()
#plt.show()
plt.savefig("Grafiken/EFeld_Messreihe_V.pdf")
plt.clf()

# Erstellen des Plots Empfindlichkeiten gegen 1/Beschleunigungsspannung

## Speichern der Empfindlichleiten
Empf_err = np.array([param_m_1, param_m_2,
                     param_m_3, param_m_4,
                     param_m_5])

## Berechnung der Regressionsgraden
popt_6, pcov_6 = curve_fit(gerade, noms(1/U_B_err), noms(Empf_err),
                           sigma=stds(Empf_err))
error_6 = np.sqrt(np.diag(pcov_6))
param_m_6 = ufloat(popt_6[0], error_6[0])
param_b_6 = ufloat(popt_6[1], error_6[1])
print("Ausgleichsgerade VI: m, b:", param_m_6, param_b_6) if PRINT else print()

## Eintragen der Messwerte
plt.errorbar(noms(1/U_B_err), noms(Empf_err),
             xerr=stds(1/U_B_err), yerr=stds(Empf_err),
             fmt="rx", label="Messwerte")

## Regressionsgerade
X = np.linspace(1.5 * 1e-03, 6.0 * 1e-03, 300)
plt.plot(X, gerade(X, *popt_6), color="gray", label="Regressionsgerade")

## Ploteinstellungen
plt.grid()
plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _:
    float(x * 1e3)))
plt.xlabel(r"Reziproke Beschleunigungsspannung $U_{B}^{-1} \ [\mathrm{mV^{-1}}]$",
           fontsize=14, family="serif")
plt.ylabel(r"Empfindlichkeit $\frac{D}{U_{d}} \ [\mathrm{\frac{m}{V}}]$",
           fontsize=14, family="serif")
plt.legend(loc="best")
plt.tight_layout()
#plt.show()
plt.savefig("Grafiken/Messreihe_VI.pdf")




# Laden der Kathodenstrahlröhren Daten
L, P, d_1, d_2 = np.loadtxt("Messdaten/Kathodendaten.txt")

# Berechnung des mittleren Abstandes

d = (d_1 + d_2)/2
d_kplx = (d_1 + (d_1 + d_2)/2)/2
d_kplx_kplx = ((d_1 * 0.55) + (((d_1 + d_2)/2) * 0.45))
print(d_kplx, d_kplx_kplx)

# Berechnung des theoretischen Vergleichswerts
theo = L * P /(2 * d)
theo_kplx = L * P /(2 * d_kplx)
theo_kplx_kplx = L * P /(2 * d_kplx_kplx)
print("Theoriewert:", theo, theo_kplx, theo_kplx_kplx)

# Vergleich des Theoriewerts mit dem berechnten
diff = np.abs(param_m_6 - theo_kplx)
diff_rel = diff / theo_kplx
diff_kplx = np.abs(param_m_6 - theo_kplx_kplx)
diff_rel_kplx = diff_kplx / theo_kplx_kplx
print("Unterschied der Ergebnisse:", diff, diff_rel, diff_kplx, diff_rel_kplx)


#==============================================================================
#
# Oszilloskop
#
#==============================================================================
#Laden der Frequenz und Verhältnisse
f_sz, n = np.loadtxt("Messdaten/Oszilloskop.txt", unpack=True)

#Berechnung der Sinusfrequenz
f_sin = f_sz * n
print("Sinusspannung:", f_sin[0], f_sin[1], f_sin[2], f_sin[3])


#==============================================================================
#
# Elektronen im magenetischen Feld
#
#==============================================================================

# Laden der Spulendaten
R_sp, N_sp = np.loadtxt("Messdaten/Spulendaten.txt")


# Berechnung des Magnetfeldes
def magenetfeld(I):
    global R_sp, N_sp
    return const.mu_0 * (8/m.sqrt(125)) * (N_sp * I/R_sp)


#==============================================================================
#
# Erdmagnetfeld
#
#==============================================================================

# Laden der Spannung, Stroms, Winkel
U_B_sp, I_hor, phi = np.loadtxt("Messdaten/Erdmagnetfeld.txt", unpack=True)

# Berechnung des horizontalen Magnetfelds
B_hor = magenetfeld(I_hor)

# Berechnung des Totalen Magnetfeldes
B_tot = B_hor * m.cos(np.deg2rad(70))

print("Totale Intensität B:", B_tot)









## Print Funktionen