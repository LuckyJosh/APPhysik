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
from sympy import *
import uncertainties as unc
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)


from aputils.utils import Quantity, ErrorEquation
from aputils.latextables.tables import Table


# Fit Funktionen
def f_exp(t, N, l):
    return N*np.exp(-l*t)


def f_gerade(t, a, b):
    return -a * t + b

#==============================================================================
class Nulleffekt:
    pass
#==============================================================================

# Laden des Nulleffekts
dt_null, N_null = np.loadtxt("Messdaten/Nulleffekt.txt")

# Zerfälle pro Sekunde
A_null = N_null/dt_null

#==============================================================================
class Rhodium:
    pass
#==============================================================================

# Laden der Messwerte
N_Rh = np.loadtxt("Messdaten/Rhodium.txt")
dt_Rh, t_Rh_max = np.loadtxt("Messdaten/Rh_Zeit.txt", unpack=True)

# Subtraktion des Nulleffekts
N_Rh_Null = np.copy(N_Rh)
N_Rh -= A_null*dt_Rh



# Berechnung der Fehler
n_Rh_err = np.sqrt(N_Rh)

# Fehlerbehafteter Messwert
N_Rh_err = unp.uarray(N_Rh, n_Rh_err)



# Plot aller Messwerte

T_Rh = np.arange(20, 740, dt_Rh)
plt.errorbar(T_Rh, np.log(N_Rh), yerr=stds(N_Rh_err)/N_Rh, fmt="rx", label="Messwerte")

plt.xlim(0,730)
plt.ylim(1.5, 5.5)
plt.xlabel(r"Zeit $t\ [s]$", fontsize=14, family="serif")
plt.ylabel(r"Logarithmierte Zerfälle $\ln(N)$", fontsize=14, family="serif")
plt.grid()
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("Grafiken/Messwerte_Rh.pdf")
#plt.show()
plt.clf()


# Wählen des Zeitpunkts t*
t_x = 480

# Regression langlebig für t >= t*
popt_Rh_2, pcov_Rh_2 = curve_fit(f_gerade, T_Rh[t_x/20:], np.log(N_Rh[t_x/20:]))
errors_Rh_2 = np.sqrt(np.diag(pcov_Rh_2))
param_a_Rh_2 = ufloat(popt_Rh_2[0], errors_Rh_2[0])
param_b_Rh_2 = ufloat(popt_Rh_2[1], errors_Rh_2[1])

print("Parameter Exp. Regression a,b :", param_a_Rh_2, param_b_Rh_2)


# Plot der Messwerte für t >= t*
plt.errorbar(T_Rh[t_x/20:], np.log(N_Rh[t_x/20:]), yerr=stds(N_Rh_err[t_x/20:])/N_Rh[t_x/20:],
             fmt="rx", label="Messwerte")

# Plot der Regressionsgeraden
T = np.linspace(0,1000,1000)
plt.plot(T, f_gerade(T, *popt_Rh_2), color="gray", label="Regressionsgerade")

plt.xlim(490,730)
plt.ylim(1.5, 4.0)
plt.xlabel(r"Zeit $t\ [s]$", fontsize=14, family="serif")
plt.ylabel(r"Logarithmierte Zerfälle $\ln(N_{l})$", fontsize=14, family="serif")
plt.grid()
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("Grafiken/Messwerte_Rh_langlebig.pdf")
#plt.show()
plt.clf()


# Subtraktion der Langlebigen Zerfälle
N_Rh_lang = np.array(np.exp(-popt_Rh_2[0]*T_Rh[:10])*exp(popt_Rh_2[1]))
#print(N_Rh_lang)

# Fehler der Werte
n_Rh_lang_err = np.array([m.sqrt(x) for x in N_Rh_lang])
# Fehlerbehaftete Größe
N_Rh_lang_err = unp.uarray(N_Rh_lang, n_Rh_lang_err)

N_Rh_kurz_err= np.array(np.subtract(N_Rh_err[:10], N_Rh_lang_err))
#print(N_Rh_kurz)
lnN_Rh_kurz = np.array([m.log(x) for x in noms(N_Rh_kurz_err)])
#print(lnN_Rh_kurz)



# Regression kurzlebig
popt_Rh_1, pcov_Rh_1 = curve_fit(f_gerade, T_Rh[:10], lnN_Rh_kurz)
errors_Rh_1 = np.sqrt(np.diag(pcov_Rh_1))
param_a_Rh_1 = ufloat(popt_Rh_1[0], errors_Rh_1[0])
param_b_Rh_1 = ufloat(popt_Rh_1[1], errors_Rh_1[1])

print("Parameter Exp. Regression a,b :", param_a_Rh_1, param_b_Rh_1)



# Plot der Messwerte für t < t*
plt.errorbar(T_Rh[:10], lnN_Rh_kurz, yerr=stds(N_Rh_kurz_err[:10])/noms(N_Rh_kurz_err)[:10],
             fmt="rx", label="Messwerte")

# Plot der Regression
T = np.linspace(0,1000,1000)
plt.plot(T, f_gerade(T, *popt_Rh_1), color="gray", label="Regressionsgerade")


plt.xlim(0,210)
plt.ylim(-1, 5)
plt.xlabel(r"Zeit $t\ [s]$", fontsize=14, family="serif")
plt.ylabel(r"Logarithmierte Zerfälle $\ln(N - N_{l})$", fontsize=14, family="serif")
plt.grid()
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("Grafiken/Messwerte_Rh_kurzlebig.pdf")
#plt.show()
plt.clf()



# Berechnung der Konstante N0(1-exp(...))
C_Rh_lang = m.exp(noms(param_b_Rh_2))
c_Rh_lang_err = m.exp(noms(param_b_Rh_2)) * stds(param_b_Rh_2)
C_Rh_lang_err = ufloat(C_Rh_lang, c_Rh_lang_err)

C_Rh_kurz = m.exp(noms(param_b_Rh_1))
c_Rh_kurz_err = m.exp(noms(param_b_Rh_1)) * stds(param_b_Rh_1)
C_Rh_kurz_err = ufloat(C_Rh_kurz, c_Rh_kurz_err)

print("Konstanten C_l, C_k:", C_Rh_lang_err.format("%.3f"), C_Rh_kurz_err.format("%.3f"))


# Berechnung der Halbwertszeit
HWZ_Rh_lang = m.log(2)/param_a_Rh_2
HWZ_Rh_kurz = m.log(2)/param_a_Rh_1

print("Halbwertzeiten", HWZ_Rh_lang.format("%.3f"), HWZ_Rh_kurz)


# Test ob  N_k << N_l gilt

N_Rh_lang = np.array(np.exp(-popt_Rh_2[0]*T_Rh[:])*exp(popt_Rh_2[1]))
N_Rh_kurz = np.array(np.exp(-popt_Rh_1[0]*T_Rh[:])*exp(popt_Rh_1[1]))


# Plot der einzelnen Zerfallskurven und Summe
#plt.plot(T_Rh, N_Rh, "xk", label="Messwerte")
plt.plot(T_Rh, N_Rh_lang, "xr", label="Zerfall mit höherer HWZ")
plt.plot(T_Rh, N_Rh_kurz, "xb", label="Zerfall mit geringerer HWZ")
plt.plot(T_Rh, N_Rh_lang + N_Rh_kurz, "xg", label="Summer beider Zerfälle")
plt.grid()
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("Grafiken/Theoriekurven.pdf")
#plt.show()
plt.clf()



######
n_Rh_Null_err = np.sqrt(N_Rh_Null)
N_Rh_Null_err = unp.uarray(N_Rh_Null, n_Rh_Null_err)

lnN_Rh = np.log(noms(N_Rh_err))
lnn_Rh_err = stds(N_Rh_err)/(N_Rh)
lnN_Rh_err = unp.uarray(lnN_Rh, lnn_Rh_err)

lnn_Rh_kurz_err = stds(N_Rh_kurz_err)/(noms(N_Rh_kurz_err))
lnN_Rh_kurz_err = unp.uarray(lnN_Rh_kurz, lnn_Rh_kurz_err)

# Tabellen
T_1 = Table(siunitx=True)
T_1.layout(seperator="column", title_row_seperator="double", border=True)
T_1.label("Auswertung_Messwerte_Rhodium")
T_1.caption("Gemessene Anzahl der Zerfäll, Anzahl der Zerfälle nach Subtraktion\
 des Nulleffekts und Werte des natürlichen Logarithmusses von diesen")
T_1.addColumn([int(t) for t in T_Rh],align="r", title="Zeit", symbol="t", unit=r"\second")
T_1.addColumn(N_Rh_Null_err,align="c", title="Zerfälle", symbol="N")
T_1.addColumn(N_Rh_err,align="c", title="Zerfälle", symbol="N - N_{0}")
T_1.addColumn(lnN_Rh_err,align="c", title="ln der Zerfälle", symbol=r"\ln(N - N_{0})")
#T_1.show()
#T_1.save("Tabellen/Messwerte_Rhodium.tex")

T_2 = Table(siunitx=True)
T_2.layout(seperator="column", title_row_seperator="double", border=True)
T_2.label("Auswertung_Messwerte_Rhodium_lang")
T_2.caption("Messwerte zur Bestimmung der Halbwertszeit des langlebigen Zerfalls für t > t*")
T_2.addColumn([int(t) for t in T_Rh[20:]],align="r", title="Zeit", symbol="t", unit=r"\second")
T_2.addColumn(N_Rh_err[20:],align="c", title="Zerfälle", symbol="N_{l}")
T_2.addColumn(lnN_Rh_err[20:],align="c", title="ln der Zerfälle", symbol=r"\ln(N - N_{0})")
#T_2.show()
#T_2.save("Tabellen/Messwerte_Rhodium_lang.tex")

T_3 = Table(siunitx=True)
T_3.layout(seperator="column", title_row_seperator="double", border=True)
T_3.label("Auswertung_Messwerte_Rhodium_kurz")
T_3.caption("Messwerte zur Bestimmung der Halbwertszeit des langlebigen Zerfalls für t << t*")
T_3.addColumn([int(t) for t in T_Rh[:10]],align="r", title="Zeit", symbol="t", unit=r"\second")
T_3.addColumn(N_Rh_err[:10],align="c", title="Zerfälle", symbol="N")
T_3.addColumn(N_Rh_kurz_err,align="c", title="Zerfälle", symbol="N - N_{l}")
T_3.addColumn(lnN_Rh_kurz_err,align="c", title="ln der Zerfälle", symbol=r"\ln(N - N_{0})")
#T_3.show()
#T_3.save("Tabellen/Messwerte_Rhodium_kurz.tex")
