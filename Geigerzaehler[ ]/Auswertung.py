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
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit
import sympy as sym
import uncertainties as unc
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)

from aputils.utils import Quantity, ErrorEquation
from aputils.latextables.tables import Table


#==============================================================================
class Allgemeines:
    pass
#==============================================================================

# Messwerte: Zeitintervalle
t_C, t_T = np.loadtxt("Messdaten/Messintervall.txt", unpack=True)

def func_gerade(x, m, b):
    return m * x + b

#==============================================================================
class Charakteristik:
    pass
#==============================================================================

# Messwerte: Spannung, Impulse, Stromstärke
U_C, P_C, I_Q = np.loadtxt("Messdaten/Charakteristik_Strom.txt", unpack=True)

# Fehlerbehaftete Messwerte
U_C_err = unp.uarray(U_C, 1)
P_C_err = unp.uarray(P_C, np.sqrt(P_C))
I_Q_err = unp.uarray(I_Q, len(I_Q)*[0.5]) * 1e-06  # Ampere

# Trennung der verwendbaren und nicht verwendbaren
U_C_n_err = np.array([U_C_err[i] for i in [4,9,10,15,17]])
U_C_j_err = np.array([U  for U in np.setdiff1d(U_C_err, U_C_n_err)])
P_C_n_err = np.array([P_C_err[i] for i in [4,9,10,15,17]])
P_C_j_err = np.array([P  for P in np.setdiff1d(P_C_err, P_C_n_err)])

#print(U_C_n_err)
#print(U_C_j_err)
#print(P_C_n_err)
#print(P_C_j_err)

# Regression der Messwerte
popt_C, pcov_C = curve_fit(func_gerade, noms(U_C_j_err),
                           noms(P_C_j_err), sigma=stds(P_C_j_err))
error = np.sqrt(np.diag(pcov_C))
param_m_C = ufloat(popt_C[0], error[0])
param_b_C = ufloat(popt_C[1], error[1])

print("Steigung:", param_m_C)
print("Steigung in % pro 100V:", param_m_C*100,"%")
print("y-Abschnitt:", param_b_C)



# Plot der Messwerte
plt.errorbar(noms(U_C_j_err), noms(P_C_j_err), yerr=stds(P_C_j_err),  fmt="rx", label="Messwerte")
plt.errorbar(noms(U_C_n_err), noms(P_C_n_err), yerr=stds(P_C_n_err),  fmt="x", color="black", alpha=0.7)

# Plot der Regression
X = np.linspace(300, 700, num=4000)
plt.plot(X, func_gerade(X, *popt_C), color="gray",
         label="Regressionsgerade")

plt.grid()
plt.xlim(300, 700)
plt.ylim(4800, 5500)
plt.xlabel("Spannung $U\ [\mathrm{V}]$")
plt.ylabel("Anzahl der Impulse $P$")

plt.legend(loc="best")
plt.savefig("Grafiken/Charakteristik.pdf")
#plt.show()

#==============================================================================
class Ladungsmenge:
    pass
#==============================================================================

Q_err = (I_Q_err * t_C)/(P_C_err)
#print(Q_err)
#print(Q_err*1e09/const.elementary_charge)

#==============================================================================
class Totzeit:
    pass
#==============================================================================

# Messwerte: Impulse 1. Quelle, 1. u. 2. Quelle und 2. Quelle, Spannung
N_1, N_12, N_2, U_T = np.loadtxt("Messdaten/Totzeitmessung.txt", unpack=True)

# Fehlerbehaftete Messwerte und Berechnung der Pulsrate
N_1_err = ufloat(N_1, np.sqrt(N_1))/t_T
N_12_err = ufloat(N_12, np.sqrt(N_12))/t_T
N_2_err = ufloat(N_2, np.sqrt(N_2))/t_T


T_err = (N_1_err + N_2_err - N_12_err)/(2*N_1_err*N_2_err)

print("Totzeit:", T_err)

#==============================================================================
class Tabellen:
    pass
#==============================================================================

Tab_C = Table(siunitx=True)
Tab_C.layout(seperator="column", title_row_seperator="double", border=True)
Tab_C.label("Auswertung_Charakteristik")
Tab_C.caption("Messwerte für die Charakteristik des Zählrohrs")
Tab_C.addColumn(U_C_err[:9], title="Spannung", symbol="U", unit=r"\volt")
Tab_C.addColumn([int(p) for p in noms(P_C_err)][:9], title="Anzahl der Pulse", symbol="P")
Tab_C.addColumn([int(p) for p in stds(P_C_err)][:9], title="Messfehler", symbol=r"\sigma_{P}")
Tab_C.addColumn(U_C_err[9:], title="Spannung", symbol="U", unit=r"\volt")
Tab_C.addColumn([int(p) for p in noms(P_C_err)][9:], title="Anzahl der Pulse", symbol="P")
Tab_C.addColumn([int(p) for p in stds(P_C_err)][9:], title="Messfehler", symbol=r"\sigma_{P}")
Tab_C.show(quiet=True)

Tab_C.save("Tabellen/Charakteristik.tex")



