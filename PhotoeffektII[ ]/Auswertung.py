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
class Allgemein:
    pass
#==============================================================================

# Messfehler: Strom, Spannung
i_err, u_err = np.loadtxt("Messdaten/Messfehler.txt", unpack=True)

def func_gerade(x, m, b):
    return m * x + b

def intercept(y, y0):
    lower = []
    greater = []
    for i in y:
        if i < y0:
            lower.append(i)
        if i > y0:
            greater.append(i)
    intercept_y = (min(greater) + max(lower))/2
    intercept_x = (X[np.where(y == min(greater))[0][0]] +
                   X[np.where(y == max(lower))[0][0]])/2
    return (intercept_x, intercept_y)



#==============================================================================
class DiodeBlau:
    pass
#==============================================================================

# Messwerte: Strom, Spannung
I_1, U_1 = np.loadtxt("Messdaten/Diode_Blau.txt", unpack=True)

# Fehlerbehaftete Messwerte: Strom, Spannung
I_1_err = unp.uarray(I_1, i_err)
U_1_err = unp.uarray(U_1, u_err)


# Regression der Messwerte
X = np.linspace(0, 5, num=50000)
popt_1, pcov_1 = curve_fit(func_gerade, noms(U_1_err[:-6]), noms(I_1_err[:-6]),
                           sigma=stds(I_1_err[:-6]))
error_1 = np.sqrt(np.diag(pcov_1))
param_m_1 = ufloat(popt_1[0], error_1[0])
param_b_1 = ufloat(popt_1[1], error_1[1])

print("Geradenparameter Blau:")
print("Steigung:{}  y-Achsenabschnitt:{}".format(param_m_1, param_b_1))


# Bestimmung der Nullstelle der Regressionsgeraden
x0_1, y0_1 = intercept(func_gerade(X, *popt_1), 0)
print("'Nullstelle' der Regressiosngeraden:", x0_1, y0_1)




# Plot der Messwerte
plt.errorbar(noms(U_1_err), noms(I_1_err), yerr=stds(I_1_err),
             fmt="rx", label="Messwerte")

# Plot der Regressionsgerade
X = np.linspace(0, 5, num=10000)
plt.plot(X, func_gerade(X, *popt_1), color="gray", label="Regressionsgerade")

# Plot der Nullstelle
plt.plot(x0_1, y0_1, "ko", alpha=0.5, label="Dispersionsspannung")


# Ploteinstellungen
plt.grid()
plt.xlim(3.2,4.8)
plt.ylim(-2,20)
plt.xlabel(r"Spannung $U\ [\mathrm{V}]$", family="serif", fontsize=14)
plt.ylabel(r"Stromstärke $I\ [\mathrm{mA}]$", family="serif", fontsize=14)
plt.legend(loc="best")
##plt.show()
plt.savefig("Grafiken/Diode_Blau.pdf")
plt.clf()



#==============================================================================
class DiodeGruen:
    pass
#==============================================================================

# Messwerte: Strom, Spannung
I_2, U_2 = np.loadtxt("Messdaten/Diode_Gruen.txt", unpack=True)

# Fehlerbehaftete Messwerte: Strom, Spannung
I_2_err = unp.uarray(I_2, i_err)
U_2_err = unp.uarray(U_2, u_err)

# Regression der Messwerte
X = np.linspace(0, 5, num=50000)
popt_2, pcov_2 = curve_fit(func_gerade, noms(U_2_err[:-2]), noms(I_2_err[:-2]),
                           sigma=stds(I_2_err[:-2]))
error_2 = np.sqrt(np.diag(pcov_2))
param_m_2 = ufloat(popt_2[0], error_2[0])
param_b_2 = ufloat(popt_2[1], error_2[1])

print("Geradenparameter Gruen:")
print("Steigung:{}  y-Achsenabschnitt:{}".format(param_m_2, param_b_2))


# Bestimmung der Nullstelle der Regressionsgeraden
x0_2, y0_2 = intercept(func_gerade(X, *popt_2), 0)
print("'Nullstelle' der Regressiosngeraden:", x0_2, y0_2)



# Plot der Regressionsgerade
X = np.linspace(0, 5, num=10000)
plt.plot(X, func_gerade(X, *popt_2), color="gray", label="Regressionsgerade")

# Plot der Nullstelle
plt.plot(x0_2, y0_2, "ko", alpha=0.5, label="Dispersionsspannung")




# Plot der Messwerte
plt.errorbar(noms(U_2_err), noms(I_2_err), yerr=stds(I_2_err),
             fmt="rx", label="Messwerte")

# Ploteinstellungen
plt.grid()
plt.xlim(1.8,2.5)
plt.ylim(-2,20)
plt.xlabel(r"Spannung $U\ [\mathrm{V}]$", family="serif", fontsize=14)
plt.ylabel(r"Stromstärke $I\ [\mathrm{mA}]$", family="serif", fontsize=14)
plt.legend(loc="best")
#plt.show()
plt.savefig("Grafiken/Diode_Gruen.pdf")
plt.clf()

#==============================================================================
class DiodeGelb:
    pass
#==============================================================================

# Messwerte: Strom, Spannung
I_3, U_3 = np.loadtxt("Messdaten/Diode_Gelb.txt", unpack=True)

# Fehlerbehaftete Messwerte: Strom, Spannung
I_3_err = unp.uarray(I_3, i_err)
U_3_err = unp.uarray(U_3, u_err)

# Regression der Messwerte
X = np.linspace(0, 5, num=50000)
popt_3, pcov_3 = curve_fit(func_gerade, noms(U_3_err[:-1]), noms(I_3_err[:-1]),
                           sigma=stds(I_3_err[:-1]))
error_3 = np.sqrt(np.diag(pcov_3))
param_m_3 = ufloat(popt_3[0], error_3[0])
param_b_3 = ufloat(popt_3[1], error_3[1])

print("Geradenparameter Gelb:")
print("Steigung:{}  y-Achsenabschnitt:{}".format(param_m_3, param_b_3))


# Bestimmung der Nullstelle der Regressionsgeraden
x0_3, y0_3 = intercept(func_gerade(X, *popt_3), 0)
print("'Nullstelle' der Regressiosngeraden:", x0_3, y0_3)


# Plot der Regressionsgerade
X = np.linspace(0, 5, num=10000)
plt.plot(X, func_gerade(X, *popt_3), color="gray", label="Regressionsgerade")

# Plot der Nullstelle
plt.plot(x0_3, y0_3, "ko", alpha=0.5, label="Dispersionsspannung")



# Plot der Messwerte
plt.errorbar(noms(U_3_err), noms(I_3_err), yerr=stds(I_3_err),
             fmt="rx", label="Messwerte")

# Ploteinstellungen
plt.grid()
plt.xlim(1.75, 2.15)
plt.ylim(-2,20)
plt.xlabel(r"Spannung $U\ [\mathrm{V}]$", family="serif", fontsize=14)
plt.ylabel(r"Stromstärke $I\ [\mathrm{mA}]$", family="serif", fontsize=14)
plt.legend(loc="best")
#plt.show()
plt.savefig("Grafiken/Diode_Gelb.pdf")
plt.clf()
#==============================================================================
class DiodeOrange:
    pass
#==============================================================================
# Messwerte: Strom, Spannung
I_4, U_4 = np.loadtxt("Messdaten/Diode_Orange.txt", unpack=True)

# Fehlerbehaftete Messwerte: Strom, Spannung
I_4_err = unp.uarray(I_4, i_err)
U_4_err = unp.uarray(U_4, u_err)


# Regression der Messwerte
X = np.linspace(0, 5, num=50000)
popt_4, pcov_4 = curve_fit(func_gerade, noms(U_4_err[:-2]), noms(I_4_err[:-2]),
                           sigma=stds(I_4_err[:-2]))
error_4 = np.sqrt(np.diag(pcov_4))
param_m_4 = ufloat(popt_4[0], error_4[0])
param_b_4 = ufloat(popt_4[1], error_4[1])

print("Geradenparameter Orange:")
print("Steigung:{}  y-Achsenabschnitt:{}".format(param_m_4, param_b_4))


# Bestimmung der Nullstelle der Regressionsgeraden
x0_4, y0_4 = intercept(func_gerade(X, *popt_4), 0)
print("'Nullstelle' der Regressiosngeraden:", x0_4, y0_4)


# Plot der Regressionsgerade
X = np.linspace(0, 5, num=10000)
plt.plot(X, func_gerade(X, *popt_4), color="gray", label="Regressionsgerade")

# Plot der Nullstelle
plt.plot(x0_4, y0_4, "ko", alpha=0.5, label="Dispersionsspannung")



# Plot der Messwerte
plt.errorbar(noms(U_4_err), noms(I_4_err), yerr=stds(I_4_err),
             fmt="rx", label="Messwerte")

# Ploteinstellungen
plt.grid()
plt.xlim(1.65, 2.10)
plt.ylim(-2,20)
plt.xlabel(r"Spannung $U\ [\mathrm{V}]$", family="serif", fontsize=14)
plt.ylabel(r"Stromstärke $I\ [\mathrm{mA}]$", family="serif", fontsize=14)
plt.legend(loc="best")
#plt.show()
plt.savefig("Grafiken/Diode_Orange.pdf")
plt.clf()
#==============================================================================
class DiodeRot:
    pass
#==============================================================================

# Messwerte: Strom, Spannung
I_5, U_5 = np.loadtxt("Messdaten/Diode_Rot.txt", unpack=True)

# Fehlerbehaftete Messwerte: Strom, Spannung
I_5_err = unp.uarray(I_5, i_err)
U_5_err = unp.uarray(U_5, u_err)


# Regression der Messwerte
X = np.linspace(0, 5, num=50000)
popt_5, pcov_5 = curve_fit(func_gerade, noms(U_5_err[:-1]), noms(I_5_err[:-1]),
                           sigma=stds(I_5_err[:-1]))
error_5 = np.sqrt(np.diag(pcov_5))
param_m_5 = ufloat(popt_5[0], error_5[0])
param_b_5 = ufloat(popt_5[1], error_5[1])

print("Geradenparameter Rot:")
print("Steigung:{}  y-Achsenabschnitt:{}".format(param_m_5, param_b_5))


# Bestimmung der Nullstelle der Regressionsgeraden
x0_5, y0_5 = intercept(func_gerade(X, *popt_5), 0)
print("'Nullstelle' der Regressiosngeraden:", x0_5, y0_5)


# Plot der Regressionsgerade
X = np.linspace(0, 5, num=10000)
plt.plot(X, func_gerade(X, *popt_5), color="gray", label="Regressionsgerade")

# Plot der Nullstelle
plt.plot(x0_5, y0_5, "ko", alpha=0.5, label="Dispersionsspannung")




# Plot der Messwerte
plt.errorbar(noms(U_5_err), noms(I_5_err), yerr=stds(I_5_err),
             fmt="rx", label="Messwerte")

# Ploteinstellungen
plt.grid()
plt.xlim(1.8, 2.3)
plt.ylim(-2,20)
plt.xlabel(r"Spannung $U\ [\mathrm{V}]$", family="serif", fontsize=14)
plt.ylabel(r"Stromstärke $I\ [\mathrm{mA}]$", family="serif", fontsize=14)
plt.legend(loc="best")
#plt.show()
plt.savefig("Grafiken/Diode_Rot.pdf")
plt.clf()

#==============================================================================
class Auswertung:
    pass
#==============================================================================
#Messwerte: Wellenlänge
wl =np.loadtxt("Messdaten/Wellenlaengen.txt")

# Berechnung der Frequenz aus der Wellenlänge
f = const.speed_of_light/wl * 1e09
print("Wellenlaenge\tFrequenz")
for i,j in zip(wl,f):
    print("{}\t\t{}".format(i,j))

# Dispersionsspannungen
U_D = [x0_1, x0_2, x0_3, x0_4, x0_5]



# Regression der Messwerte
popt_6, pcov_6 = curve_fit(func_gerade, f, U_D)
error_6 = np.sqrt(np.diag(pcov_6))
param_m_6 = ufloat(popt_6[0], error_6[0])
param_b_6 = ufloat(popt_6[1], error_6[1])

print("Geradenparameter Dispersionsspannung:")
print("Steigung:{}  y-Achsenabschnitt:{}".format(param_m_6, param_b_6))


# Plot der Regressionsgeraden
X = np.linspace(0, 1e15, num=5e4)
plt.plot(X, func_gerade(X, *popt_6), color="gray", label="Regressionsgerade")

# Plot der Dispersionsspannung gegen die Frequenz
plt.plot(f, U_D, "rx", label="Messwerte")

# Ploteinstellungen
plt.grid()
plt.xlim(0.4e15,0.7e15)
plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x,_ : float(x)*1e-15))
plt.ylim(0,4)
plt.ylabel(r"Dispersiosspannung $U_{D}\ [\mathrm{V}]$", family="serif", fontsize=14)
plt.xlabel(r"Frequenz $f\ [\mathrm{PHz}]$", family="serif", fontsize=14)
plt.legend(loc="best")
#plt.show()
plt.savefig("Grafiken/Dispersion_Frequenz.pdf")
plt.clf()


#==============================================================================
class Tabellen:
    pass
#==============================================================================

# Blau
Tab = Table(siunitx=True)
Tab.layout(seperator="column", title_row_seperator="double", border=True)
Tab.caption(r"Messwerte der Spannung und des Stroms für die blaue Diode mit der Wellenlänge \SI{465}{\nm}")
Tab.label("Auswertung_Diode_Blau")

Tab.addColumn(U_1_err[:10], title="Spannung", symbol="U", unit=r"\volt")
Tab.addColumn(I_1_err[:10], title="Strom", symbol="I", unit=r"\ampere")
Tab.addColumn(U_1_err[10:], title="Spannung", symbol="U", unit=r"\volt")
Tab.addColumn(I_1_err[10:], title="Strom", symbol="I", unit=r"\ampere")
#Tab.show()
#Tab.save("Tabellen/Diode_Blau.tex")

# Gruen
Tab = Table(siunitx=True)
Tab.layout(seperator="column", title_row_seperator="double", border=True)
Tab.caption(r"Messwerte der Spannung und des Stroms für die grüne Diode mit der Wellenlänge \SI{565}{\nm} ")
Tab.label("Auswertung_Diode_Gruen")
Tab.addColumn(U_2_err[:6], title="Spannung", symbol="U", unit=r"\volt")
Tab.addColumn(I_2_err[:6], title="Strom", symbol="I", unit=r"\ampere")
Tab.addColumn(U_2_err[6:], title="Spannung", symbol="U", unit=r"\volt")
Tab.addColumn(I_2_err[6:], title="Strom", symbol="I", unit=r"\ampere")
#Tab.show()
#Tab.save("Tabellen/Diode_Gruen.tex")

# Gelb
Tab = Table(siunitx=True)
Tab.layout(seperator="column", title_row_seperator="double", border=True)
Tab.caption(r"Messwerte der Spannung und des Stroms für die gelbe Diode  mit der Wellenlänge \SI{585}{\nm} ")
Tab.label("Auswertung_Diode_Gelb")
Tab.addColumn(U_3_err[:5], title="Spannung", symbol="U", unit=r"\volt")
Tab.addColumn(I_3_err[:5], title="Strom", symbol="I", unit=r"\ampere")
Tab.addColumn(U_3_err[5:], title="Spannung", symbol="U", unit=r"\volt")
Tab.addColumn(I_3_err[5:], title="Strom", symbol="I", unit=r"\ampere")
#Tab.show()
#Tab.save("Tabellen/Diode_Gelb.tex")

# Orange
Tab = Table(siunitx=True)
Tab.layout(seperator="column", title_row_seperator="double", border=True)
Tab.caption(r"Messwerte der Spannung und des Stroms für die orangefarbene Diode  mit der Wellenlänge \SI{635}{\nm} ")
Tab.label("Auswertung_Diode_Orange")
Tab.addColumn(U_4_err[:6], title="Spannung", symbol="U", unit=r"\volt")
Tab.addColumn(I_4_err[:6], title="Strom", symbol="I", unit=r"\ampere")
Tab.addColumn(U_4_err[6:], title="Spannung", symbol="U", unit=r"\volt")
Tab.addColumn(I_4_err[6:], title="Strom", symbol="I", unit=r"\ampere")
#Tab.show()
#Tab.save("Tabellen/Diode_Orange.tex")

# Rot
Tab = Table(siunitx=True)
Tab.layout(seperator="column", title_row_seperator="double", border=True)
Tab.caption(r"Messwerte der Spannung und des Stroms für die roten Diode  mit der Wellenlänge \SI{657}{\nm} ")
Tab.label("Auswertung_Diode_Rot")
Tab.addColumn(U_5_err[:6], title="Spannung", symbol="U", unit=r"\volt")
Tab.addColumn(I_5_err[:6], title="Strom", symbol="I", unit=r"\ampere")
Tab.addColumn(U_5_err[6:], title="Spannung", symbol="U", unit=r"\volt")
Tab.addColumn(I_5_err[6:], title="Strom", symbol="I", unit=r"\ampere")
#Tab.show()
#Tab.save("Tabellen/Diode_Rot.tex")


# Regressionsparameter
param_m = [param_m_1, param_m_2, param_m_3, param_m_4, param_m_5]
param_b = [param_b_1, param_b_2, param_b_3, param_b_4, param_b_5]



Tab = Table(siunitx=True)
Tab.layout(seperator="column", title_row_seperator="double", border=True)
Tab.caption(r"Regressionsparameter für die jeweils angegbenen Wellenlängen und Frequenzen")
Tab.label("Auswertung_Parameter")
Tab.addColumn([int(l) for l in wl], title="Wellenlänge", symbol="\lambda", unit=r"\nm")
Tab.addColumn(f*1e-15, title="Frequenz", symbol="f", unit=r"\peta\hertz")
Tab.addColumn(param_m, title="Steigung", symbol="m", unit=r"\milli\ampere\per\volt")
Tab.addColumn(param_b, title="y-Achsenabschnitt", symbol="b", unit=r"\milli\ampere")

#Tab.show()
#Tab.save("Tabellen/Dioden_Regression_Parameter.tex")

