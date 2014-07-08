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
import sys

from aputils.utils import Quantity, ErrorEquation, OutputFile
from aputils.latextables.tables import Table

sys.stdout = OutputFile("Daten/log.txt")

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
mI_Q_err = unp.uarray(I_Q, len(I_Q)*[0.1])  # microAmpere
I_Q_err = unp.uarray(I_Q, len(I_Q)*[0.1]) * 1e-06 # microAmpere

# Werte sollten nicht rausgenommen werden
## Trennung der verwendbaren und nicht verwendbaren
#U_C_n_err = np.array([U_C_err[i] for i in [4,9,10,15,17]])
#U_C_j_err = np.array([U  for U in np.setdiff1d(U_C_err, U_C_n_err)])
#P_C_n_err = np.array([P_C_err[i] for i in [4,9,10,15,17]])
#P_C_j_err = np.array([P  for P in np.setdiff1d(P_C_err, P_C_n_err)])
#
##print(U_C_n_err)
##print(U_C_j_err)
##print(P_C_n_err)
##print(P_C_j_err)
#
## Regression der Messwerte
#popt_C, pcov_C = curve_fit(func_gerade, noms(U_C_j_err),
#                           noms(P_C_j_err), sigma=stds(P_C_j_err))
#error = np.sqrt(np.diag(pcov_C))
#param_m_C = ufloat(popt_C[0], error[0])
#param_b_C = ufloat(popt_C[1], error[1])
#
#print("Steigung:", param_m_C)
#print("Steigung in % pro 100V:", param_m_C*100,"%")
#print("y-Abschnitt:", param_b_C)

# Regression der Messwerte
popt_C, pcov_C = curve_fit(func_gerade, noms(U_C_err),
                           noms(P_C_err), sigma=stds(P_C_err))
error = np.sqrt(np.diag(pcov_C))
param_m_C = ufloat(popt_C[0], error[0])
param_b_C = ufloat(popt_C[1], error[1])

print("Steigung:", param_m_C)
print(func_gerade(100, param_m_C,param_b_C))
print(func_gerade(0, param_m_C,param_b_C))
print((func_gerade(100, param_m_C,param_b_C)- func_gerade(0, param_m_C,param_b_C))/func_gerade(0, param_m_C,param_b_C))
print("Steigung in % pro 100V:", 100 * (func_gerade(100, param_m_C,param_b_C)- func_gerade(0, param_m_C,param_b_C))/func_gerade(0, param_m_C,param_b_C),"%")
print("y-Abschnitt:", param_b_C)



# Plot der Messwerte
plt.errorbar(noms(U_C_err), noms(P_C_err), yerr=stds(P_C_err),  fmt="rx", label="Messwerte")


# Plot der Regression
X = np.linspace(300, 700, num=4000)
plt.plot(X, func_gerade(X, *popt_C), color="gray",
         label="Regressionsgerade")

plt.grid()
plt.xlim(300, 700)
plt.ylim(4800, 5500)
plt.xlabel("Spannung $U\ [\mathrm{V}]$")
plt.ylabel("Anzahl der Impulse $Z$")

plt.legend(loc="best")
plt.savefig("Grafiken/Charakteristik.pdf")
#plt.show()

#==============================================================================
class Ladungsmenge:
    pass
#==============================================================================

Q_err = (I_Q_err * t_C)/(P_C_err)
print(Q_err)
Q_err /= const.elementary_charge*1e09
#for q in Q_err:
#    print(int(noms(q)),"(",int(stds(q)),")")
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
#Tab_C.show(quiet=True)

#Tab_C.save("Tabellen/Charakteristik.tex")

Tab_T = Table(siunitx=True)
Tab_T.layout(seperator="column", title_row_seperator="double", border=True)
Tab_T.label("Auswertung_Totzeit")
Tab_T.caption("Aufgenommene Impulsraten der Einzelquellen und beider Quellen")
Tab_T.addColumn([N_1_err], title="Impulsrate 1", symbol="N_1", unit=r"\per\second")
Tab_T.addColumn([N_2_err], title="Impulsrate 2", symbol="N_2", unit=r"\per\second")
Tab_T.addColumn([N_12_err], title="Impulsrate 1+2", symbol="N_{1+2}", unit=r"\per\second")
#Tab_T.show(quiet=True)
#Tab_T.save("Tabellen/Totzeit.tex")

Tab_Q = Table(siunitx=True)
Tab_Q.layout(seperator="column", title_row_seperator="double", border=True)
Tab_Q.label("Auswertung_Ladungsmenge")
Tab_Q.caption("Aufgenommene Stromstärken und Impulsraten zu der jeweilig anliegenden Spannung")
Tab_Q.addColumn(U_C_err[:9], title="Spannung", symbol="U", unit=r"\volt")
Tab_Q.addColumn(mI_Q_err[:9], title="Stromstärke", symbol=r"\overline{I}", unit=r"\micro\ampere")
Tab_Q.addColumn(P_C_err[:9]/t_C, title="Impulserate", symbol="N", unit=r"\per\second")
Tab_Q.addColumn(Q_err[:9], title="Ladungsmenge", symbol="\Delta Q", unit=r"\giga e")

Tab_Q.addColumn(U_C_err[9:], title="Spannung", symbol="U", unit=r"\volt")
Tab_Q.addColumn(mI_Q_err[9:], title="Stromstärke", symbol=r"\overline{I}", unit=r"\micro\ampere")
Tab_Q.addColumn(P_C_err[9:]/t_C, title="Impulsrate", symbol="N", unit=r"\per\second")
Tab_Q.addColumn(Q_err[9:], title="Ladungsmenge", symbol="\Delta Q", unit=r"\giga e")

#Tab_Q.show(quiet=True)
#Tab_Q.save("Tabellen/Ladungsmenge.tex")
#==============================================================================
class Fehlergleichungen:
    pass
#==============================================================================

N1, N2, N12 = sym.var("N_1, N_2, N_{1+2}")
T = (N1 + N2 - N12)/2*N1*N2
T_Eq = ErrorEquation(T, name="T")
print(T_Eq.std)




