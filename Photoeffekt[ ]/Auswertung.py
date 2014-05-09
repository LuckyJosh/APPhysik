
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
import sys
from sympy import *
import uncertainties as unc
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)


from aputils.utils import Quantity, ErrorEquation, OutputFile
from aputils.latextables.tables import Table

sys.stdout = OutputFile("Daten/Ausgabe.txt")


SHOW = False


def gerade(x, m, b):
    return m * x + b



def schnittpunkt(y, y0):
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
class orange:
    pass
#==============================================================================
# Laden der Messwerte
I_1, U_1 = np.loadtxt("Messdaten/Orange.txt", unpack=True)

# Ladend er Fehler
I_err, U_err = np.loadtxt("Messdaten/Fehler.txt", unpack=True)

# fehlerbehaftete Größen
I_1_err = unp.uarray(I_1, len(I_1)*[I_err])
U_1_err = unp.uarray(U_1, len(U_1)*[U_err])

# Wurzel des Stroms
Iw_1 = np.sqrt(noms(I_1_err[1:]))
iw_1_err = (0.5 * stds(I_1_err[1:])/Iw_1)
Iw_1_err = unp.uarray(Iw_1, iw_1_err)

#iw_1_err = (0.5 * stds(I_1_err[1:])/Iw_1[1:])
#Iw_1_err = unp.uarray(Iw_1[1:], iw_1_err)
#Iw_1_err = np.concatenate((unp.uarray([0], [0]), Iw_1_err))



# Regression
popt_1, pcov_1 = curve_fit(gerade, noms(U_1_err[1:]), noms(Iw_1_err),
                           sigma=stds(Iw_1_err))
errors_1 = np.sqrt(np.diag(pcov_1))
param_1_m = ufloat(popt_1[0], errors_1[0])
param_1_b = ufloat(popt_1[1], errors_1[1])

print("Regressionsparameter Orange:", param_1_m, param_1_b)

# Einstellungen
plt.grid()
plt.xlabel(r"Bremsspannung $\frac{U}{\mathrm{V}}$", fontsize=14, family="serif")
plt.ylabel(r"Photostrom $\sqrt{\frac{I}{\mathrm{pA}}}$", fontsize=14, family="serif")
plt.xlim(0.0, 0.4)
plt.ylim(-0.5, 4)


# Messwerte
plt.errorbar(noms(U_1_err[1:]), noms(Iw_1_err), xerr=stds(U_1_err[1:]),
             yerr=stds(Iw_1_err),
             label="Messwerte", fmt="x", color="red")


# Regressionsgerade
X = np.linspace(0.0, 0.4, num=10000)
plt.plot(X, gerade(X, *popt_1), label="Regressionsgerade", color="gray")

# Nullstelle
x_0_1 = schnittpunkt(gerade(X, *popt_1), 0)[0]
plt.plot(x_0_1, 0, "o", label="Grenzspannung", color="black", alpha=0.7)
print("Nullstelle:", x_0_1)

# Einstellungen2
plt.legend(loc="best")
plt.tight_layout()
plt.show() if SHOW else plt.savefig("Grafiken/Orange.pdf")
plt.clf()

# Tabelle
Tab_1 = Table(siunitx=True)
Tab_1.caption("Messwerte der orangenen Spektrallinie")
Tab_1.label("Messwerte_Orange")
Tab_1.layout(seperator="column", title_row_seperator="double", border=True)
Tab_1.addColumn(I_1_err[:5], title="Photostrom", symbol="I", unit=r"\pico\ampere")
Tab_1.addColumn(U_1_err[:5], title="Bremsspannung", symbol="U", unit=r"\volt")
Tab_1.addColumn(I_1_err[5:], title="Photostrom", symbol="I", unit=r"\pico\ampere")
Tab_1.addColumn(U_1_err[5:], title="Bremsspannung", symbol="U", unit=r"\volt")
#Tab_1.show()
#Tab_1.save("Tabellen/Messwerte_Orange.tex")

#==============================================================================
class gruen:
    pass
#==============================================================================
# Laden der Messwerte
I_2, U_2 = np.loadtxt("Messdaten/Gruen.txt", unpack=True)

# Ladend er Fehler
I_err, U_err = np.loadtxt("Messdaten/Fehler.txt", unpack=True)

# fehlerbehaftete Größen
I_2_err = unp.uarray(I_2, len(I_2)*[I_err])
U_2_err = unp.uarray(U_2, len(U_2)*[U_err])

# Wurzel des Stroms
Iw_2 = np.sqrt(noms(I_2_err[1:]))
iw_2_err = (0.5 * stds(I_2_err[1:])/Iw_2)
Iw_2_err = unp.uarray(Iw_2, iw_2_err)
#iw_2_err = (0.5 * stds(I_2_err[1:])/Iw_2[1:])
#Iw_2_err = unp.uarray(Iw_2[1:], iw_2_err)
#Iw_2_err = np.concatenate((unp.uarray([0], [0]), Iw_2_err))

# Regression
popt_2, pcov_2 = curve_fit(gerade, noms(U_2_err[1:]), noms(Iw_2_err),
                           sigma=stds(Iw_2_err))
errors_2 = np.sqrt(np.diag(pcov_2))
param_2_m = ufloat(popt_2[0], errors_2[0])
param_2_b = ufloat(popt_2[1], errors_2[1])

print("Regressionsparameter Gruen:", param_2_m, param_2_b)

# Einstellungen
plt.grid()
plt.xlabel(r"Bremsspannung $\frac{U}{\mathrm{V}}$", fontsize=14, family="serif")
plt.ylabel(r"Photostrom $\sqrt{\frac{I}{\mathrm{pA}}}$", fontsize=14, family="serif")
plt.xlim(-0.1, 0.5)
plt.ylim(-0.5, 9)


# Messwerte
plt.errorbar(noms(U_2_err[1:]), noms(Iw_2_err), xerr=stds(U_2_err[1:]),
             yerr=stds(Iw_2_err),
             label="Messwerte", fmt="x", color="red")


# Regressionsgerade
X = np.linspace(-1, 1, num=10000)
plt.plot(X, gerade(X, *popt_2), label="Regressionsgerade", color="gray")

# Nullstelle
x_0_2 = schnittpunkt(gerade(X, *popt_2), 0)[0]
plt.plot(x_0_2, 0, "o", label="Grenzspannung", color="black", alpha=0.7)
print("Nullstelle:", x_0_2)

# Einstellungen2
plt.legend(loc="best")
plt.tight_layout()
plt.show() if SHOW else plt.savefig("Grafiken/Gruen.pdf")
plt.clf()
# Tabelle
Tab_2 = Table(siunitx=True)
Tab_2.caption("Messwerte der grünen Spektrallinie")
Tab_2.label("Messwerte_Gruen")
Tab_2.layout(seperator="column", title_row_seperator="double", border=True)
Tab_2.addColumn(I_2_err[:8], title="Photostrom", symbol="I", unit=r"\pico\ampere")
Tab_2.addColumn(U_2_err[:8], title="Bremsspannung", symbol="U", unit=r"\volt")
Tab_2.addColumn(np.append(I_2_err[8:], 0), title="Photostrom", symbol="I", unit=r"\pico\ampere")
Tab_2.addColumn(np.append(U_2_err[8:], 0), title="Bremsspannung", symbol="U", unit=r"\volt")
#Tab_2.show()
#Tab_2.save("Tabellen/Messwerte_Gruen.tex")
#==============================================================================
class cyan:
    pass
#==============================================================================
# Laden der Messwerte
I_3, U_3 = np.loadtxt("Messdaten/Cyan.txt", unpack=True)

# Ladend er Fehler
I_err, U_err = np.loadtxt("Messdaten/Fehler.txt", unpack=True)

# fehlerbehaftete Größen
I_3_err = unp.uarray(I_3, len(I_3)*[I_err])
U_3_err = unp.uarray(U_3, len(U_3)*[U_err])

# Wurzel des Stroms
Iw_3 = np.sqrt(noms(I_3_err[1:]))
iw_3_err = (0.5 * stds(I_3_err[1:])/Iw_3)
Iw_3_err = unp.uarray(Iw_3, iw_3_err)

#iw_3_err = (0.5 * stds(I_3_err[1:])/Iw_3[1:])
#Iw_3_err = unp.uarray(Iw_3[1:], iw_3_err)
#Iw_3_err = np.concatenate((unp.uarray([0], [0]), Iw_3_err))

# Regression
popt_3, pcov_3 = curve_fit(gerade, noms(U_3_err[1:]), noms(Iw_3_err),
                           sigma=stds(Iw_3_err))
errors_3 = np.sqrt(np.diag(pcov_3))
param_3_m = ufloat(popt_3[0], errors_3[0])
param_3_b = ufloat(popt_3[1], errors_3[1])

print("Regressionsparameter Cyan:", param_3_m, param_3_b)

# Einstellungen
plt.grid()
plt.xlabel(r"Bremsspannung $\frac{U}{\mathrm{V}}$", fontsize=14, family="serif")
plt.ylabel(r"Photostrom $\sqrt{\frac{I}{\mathrm{pA}}}$", fontsize=14, family="serif")
plt.xlim(-0.1, 0.8)
plt.ylim(-0.5, 3)


# Messwerte
plt.errorbar(noms(U_3_err[1:]), noms(Iw_3_err), xerr=stds(U_3_err[1:]),
             yerr=stds(Iw_3_err),
             label="Messwerte", fmt="x", color="red")


# Regressionsgerade
X = np.linspace(-1, 1, num=10000)
plt.plot(X, gerade(X, *popt_3), label="Regressionsgerade", color="gray")

# Nullstelle
x_0_3 = schnittpunkt(gerade(X, *popt_3), 0)[0]
plt.plot(x_0_3, 0, "o", label="Grenzspannung", color="black", alpha=0.7)
print("Nullstelle:", x_0_3)

# Einstellungen2
plt.legend(loc="best")
plt.tight_layout()
plt.show() if SHOW else plt.savefig("Grafiken/Cyan.pdf")
plt.clf()
# Tabelle
Tab_3 = Table(siunitx=True)
Tab_3.caption("Messwerte der cyanen Spektrallinie")
Tab_3.label("Messwerte_Cyan")
Tab_3.layout(seperator="column", title_row_seperator="double", border=True)
Tab_3.addColumn(I_3_err[:5], title="Photostrom", symbol="I", unit=r"\pico\ampere")
Tab_3.addColumn(U_3_err[:5], title="Bremsspannung", symbol="U", unit=r"\volt")
Tab_3.addColumn(I_3_err[5:], title="Photostrom", symbol="I", unit=r"\pico\ampere")
Tab_3.addColumn(U_3_err[5:], title="Bremsspannung", symbol="U", unit=r"\volt")
#Tab_3.show()
#Tab_3.save("Tabellen/Messwerte_Cyan.tex")
#==============================================================================
class violett1:
    pass
#==============================================================================
# Laden der Messwerte
I_4, U_4 = np.loadtxt("Messdaten/Violett_1.txt", unpack=True)

# Ladend er Fehler
I_err, U_err = np.loadtxt("Messdaten/Fehler.txt", unpack=True)

# fehlerbehaftete Größen
I_4_err = unp.uarray(I_4, len(I_4)*[I_err])
U_4_err = unp.uarray(U_4, len(U_4)*[U_err])

# Wurzel des Stroms
Iw_4 = np.sqrt(noms(I_4_err[1:]))
iw_4_err = (0.5 * stds(I_4_err[1:])/Iw_4)
Iw_4_err = unp.uarray(Iw_4, iw_4_err)

#iw_4_err = (0.5 * stds(I_4_err[1:])/Iw_4[1:])
#Iw_4_err = unp.uarray(Iw_4[1:], iw_4_err)
#Iw_4_err = np.concatenate((unp.uarray([0], [0]), Iw_4_err))

# Regression
popt_4, pcov_4 = curve_fit(gerade, noms(U_4_err[1:]), noms(Iw_4_err),
                           sigma=stds(Iw_4_err))
errors_4 = np.sqrt(np.diag(pcov_4))
param_4_m = ufloat(popt_4[0], errors_4[0])
param_4_b = ufloat(popt_4[1], errors_4[1])

print("Regressionsparameter Violett1:", param_4_m, param_4_b)

# Einstellungen
plt.grid()
plt.xlabel(r"Bremsspannung $\frac{U}{\mathrm{V}}$", fontsize=14, family="serif")
plt.ylabel(r"Photostrom $\sqrt{\frac{I}{\mathrm{pA}}}$", fontsize=14, family="serif")
plt.xlim(-0.1, 1)
plt.ylim(-0.5, 20)


# Messwerte
plt.errorbar(noms(U_4_err[1:]), noms(Iw_4_err), xerr=stds(U_4_err[1:]),
             yerr=stds(Iw_4_err),
             label="Messwerte", fmt="x", color="red")


# Regressionsgerade
X = np.linspace(-1, 1, num=10000)
plt.plot(X, gerade(X, *popt_4), label="Regressionsgerade", color="gray")

# Nullstelle
x_0_4 = schnittpunkt(gerade(X, *popt_4), 0)[0]
plt.plot(x_0_4, 0, "o", label="Grenzspannung", color="black", alpha=0.7)
print("Nullstelle:", x_0_4)

# Einstellungen2
plt.legend(loc="best")
plt.tight_layout()
plt.show() if SHOW else plt.savefig("Grafiken/Violett1.pdf")
plt.clf()


# Tabelle
Tab_4 = Table(siunitx=True)
Tab_4.caption("Messwerte der ersten violetten Spektrallinie")
Tab_4.label("Messwerte_Violett1")
Tab_4.layout(seperator="column", title_row_seperator="double", border=True)
Tab_4.addColumn(I_4_err[:5], title="Photostrom", symbol="I", unit=r"\pico\ampere")
Tab_4.addColumn(U_4_err[:5], title="Bremsspannung", symbol="U", unit=r"\volt")
Tab_4.addColumn(I_4_err[5:], title="Photostrom", symbol="I", unit=r"\pico\ampere")
Tab_4.addColumn(U_4_err[5:], title="Bremsspannung", symbol="U", unit=r"\volt")
#Tab_4.show()
#Tab_4.save("Tabellen/Messwerte_Violett1.tex")
#==============================================================================
class violett2:
    pass
#==============================================================================
# Laden der Messwerte
I_5, U_5 = np.loadtxt("Messdaten/Violett_2.txt", unpack=True)

# Ladend er Fehler
I_err, U_err = np.loadtxt("Messdaten/Fehler.txt", unpack=True)

# fehlerbehaftete Größen
I_5_err = unp.uarray(I_5, len(I_5)*[I_err])
U_5_err = unp.uarray(U_5, len(U_5)*[U_err])

# Wurzel des Stroms
Iw_5 = np.sqrt(noms(I_5_err[1:]))
iw_5_err = (0.5 * stds(I_5_err[1:])/Iw_5)
Iw_5_err = unp.uarray(Iw_5, iw_5_err)

#iw_5_err = (0.5 * stds(I_5_err[1:])/Iw_5[1:])
#Iw_5_err = unp.uarray(Iw_5[1:], iw_5_err)
#Iw_5_err = np.concatenate((unp.uarray([0], [0]), Iw_5_err))

# Regression
popt_5, pcov_5 = curve_fit(gerade, noms(U_5_err[1:]), noms(Iw_5_err),
                           sigma=stds(Iw_5_err))
errors_5 = np.sqrt(np.diag(pcov_5))
param_5_m = ufloat(popt_5[0], errors_5[0])
param_5_b = ufloat(popt_5[1], errors_5[1])

print("Regressionsparameter Violett2:", param_5_m, param_5_b)

# Einstellungen
plt.grid()
plt.xlabel(r"Bremsspannung $\frac{U}{\mathrm{V}}$", fontsize=14, family="serif")
plt.ylabel(r"Photostrom $\sqrt{\frac{I}{\mathrm{pA}}}$", fontsize=14, family="serif")
plt.xlim(-0.1, 1.2)
plt.ylim(-0.5, 10)


# Messwerte
plt.errorbar(noms(U_5_err[1:]), noms(Iw_5_err), xerr=stds(U_5_err[1:]),
             yerr=stds(Iw_5_err),
             label="Messwerte", fmt="x", color="red")


# Regressionsgerade
X = np.linspace(-1, 2, num=10000)
plt.plot(X, gerade(X, *popt_5), label="Regressionsgerade", color="gray")

# Nullstelle
x_0_5 = schnittpunkt(gerade(X, *popt_5), 0)[0]
plt.plot(x_0_5, 0, "o", label="Grenzspannung", color="black", alpha=0.7)
print("Nullstelle:", x_0_5)

# Einstellungen2
plt.legend(loc="best")
plt.tight_layout()
plt.show() if SHOW else plt.savefig("Grafiken/Violett2.pdf")
plt.clf()

# Tabelle
Tab_5 = Table(siunitx=True)
Tab_5.caption("Messwerte der zweiten violetten Spektrallinie")
Tab_5.label("Messwerte_Violett2")
Tab_5.layout(seperator="column", title_row_seperator="double", border=True)
Tab_5.addColumn(I_5_err[:5], title="Photostrom", symbol="I", unit=r"\pico\ampere")
Tab_5.addColumn(U_5_err[:5], title="Bremsspannung", symbol="U", unit=r"\volt")
Tab_5.addColumn(I_5_err[5:], title="Photostrom", symbol="I", unit=r"\pico\ampere")
Tab_5.addColumn(U_5_err[5:], title="Bremsspannung", symbol="U", unit=r"\volt")
#Tab_5.show()
#Tab_5.save("Tabellen/Messwerte_Violett2.tex")




#==============================================================================
# Alle Parameter
#==============================================================================
Param_M = np.array([param_1_m, param_2_m, param_3_m, param_4_m, param_5_m,])
Param_B = np.array([param_1_b, param_2_b, param_3_b, param_4_b, param_5_b,])
wl = np.loadtxt("Messdaten/Wellenlaengen.txt")
f = const.speed_of_light/(wl * 1e-09)
U_g = np.array([x_0_1, x_0_2, x_0_3, x_0_4, x_0_5])

Tab_param = Table(siunitx=True)
Tab_param.caption("Regressionsparameter der Untersuchung der Spektrallinien")
Tab_param.label("Messwerte_ParameterSpektrum")
Tab_param.layout(seperator="column", title_row_seperator="double", border=True)
Tab_param.addColumn([int(w) for w in wl], title="Wellenlänge",
                 symbol=r"\lambda", unit=r"\nano\meter")
Tab_param.addColumn(f*1e-15, title="Frequenz", symbol="f", unit=r"\peta\hertz")
Tab_param.addColumn(Param_M, title="Steigung", symbol="a")
Tab_param.addColumn(Param_B, title="y-Achsenabschnitt", symbol="b")
Tab_param.addColumn(U_g, title="Grenzspannung", symbol=r"U_{g}", unit=r"\volt")
#Tab_param.show()
#Tab_param.save("Tabellen/Messwerte_Parameter.tex")

#==============================================================================
class orange2:
    pass
#==============================================================================
# Laden der Messwerte
I_6, U_6 = np.loadtxt("Messdaten/Messung_2.txt", unpack=True)

# Ladend er Fehler
I_err, U_err = np.loadtxt("Messdaten/Fehler.txt", unpack=True)

# fehlerbehaftete Größen
I_6_err = unp.uarray(I_6, len(I_6)*[I_err])
U_6_err = -1*unp.uarray(U_6, len(U_6)*[U_err])


## Regression
#popt_6, pcov_6 = curve_fit(gerade, noms(U_6_err), noms(Iw_6_err),
#                           sigma=stds(Iw_6_err))
#errors_6 = np.sqrt(np.diag(pcov_1))
#param_6_m = ufloat(popt_6[0], errors_6[0])
#param_6_b = ufloat(popt_6[1], errors_6[1])

#print("Regressionsparameter Violett2:", param_6_m, param_6_b)

# Einstellungen
plt.grid()
plt.xlabel(r"Bremsspannung $\frac{U}{\mathrm{V}}$", fontsize=14, family="serif")
plt.ylabel(r"Photostrom $\frac{I}{\mathrm{pA}}$", fontsize=14, family="serif")
#plt.xlim(-20, 1.2)
#plt.ylim(-0.5, 10)


# Messwerte
plt.errorbar(noms(U_6_err), noms(I_6_err), xerr=stds(U_6_err),
             yerr=stds(I_6_err),
             label="Messwerte", fmt="x", color="red")


# Regressionsgerade
#X = np.linspace(-1, 2, num=5500)
#plt.plot(X, gerade(X, *popt_5), label="Regressionsgerade", color="gray")


# Einstellungen2
plt.legend(loc="best")
plt.tight_layout()
plt.show() if SHOW else plt.savefig("Grafiken/Orange2.pdf")
plt.clf()
# Tabelle
Tab_6 = Table(siunitx=True)
Tab_6.caption("Messwerte der orangenen Spektrallinie bei verschiedenen Bremsspannungen")
Tab_6.label("Messwerte_Messung2")
Tab_6.layout(seperator="column", title_row_seperator="double", border=True)
Tab_6.addColumn(I_6_err[:20], title="Photostrom", symbol="I", unit=r"\pico\ampere")
Tab_6.addColumn(U_6_err[:20], title="Bremsspannung", symbol="U", unit=r"\volt")
Tab_6.addColumn(I_6_err[20:], title="Photostrom", symbol="I", unit=r"\pico\ampere")
Tab_6.addColumn(U_6_err[20:], title="Bremsspannung", symbol="U", unit=r"\volt")
#Tab_6.show()
#Tab_6.save("Tabellen/Messwerte_Messung2.tex")
#==============================================================================
class bunt:
    pass
#==============================================================================
U_g = np.array([x_0_1, x_0_2, x_0_3, x_0_4, x_0_5])
wl = np.loadtxt("Messdaten/Wellenlaengen.txt")
f = const.speed_of_light/(wl * 1e-09)

# Regression
popt_7, pcov_7 = curve_fit(gerade, f, U_g)
errors_7 = np.sqrt(np.diag(pcov_7))
param_7_m = ufloat(popt_7[0], errors_7[0])
param_7_b = ufloat(popt_7[1], errors_7[1])
theo = const.h/const.elementary_charge
print("Regressionsparameter Bunt:", param_7_m, param_7_b)
print("Regressionsparameter Bunt:", param_7_m/const.elementary_charge, param_7_b)
print("Theoriewert, Abweichung:", theo, (param_7_m-theo)/theo)
# Einstellungen
plt.grid()
plt.ylabel(r"Bremsspannung $\frac{U}{\mathrm{V}}$", fontsize=14, family="serif")
plt.xlabel(r"Frequenz $\frac{f}{\mathrm{PHz}}$", fontsize=14, family="serif")
plt.xlim(0.4e15, 0.8e15)
plt.ylim(0, 1.5)
plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: x * 1e-15))


# Messwerte
plt.errorbar(f, U_g, label="Messwerte", fmt="x", color="red")

# Regressionsgerade
X = np.linspace(-1e15, 1e15, num=100000)
plt.plot(X, gerade(X, *popt_7), label="Regressionsgerade", color="gray")


# Einstellungen2
plt.legend(loc="best")
plt.tight_layout()
plt.show() if SHOW else plt.savefig("Grafiken/Messreihe2.pdf")
plt.clf()
# Tabelle
Tab_7 = Table(siunitx=True)
Tab_7.caption("Bestimmte Grenzspannungen mit zugehöriger Wellenlänge bzw. Frequenz")
Tab_7.label("Messwerte_Bunt")
Tab_7.layout(seperator="column", title_row_seperator="double", border=True)
Tab_7.addColumn([int(w) for w in wl], title="Wellenlänge", symbol=r"\lambda", unit=r"\nano\meter")
Tab_7.addColumn(f*1e-15, title="Frequenz", symbol="f", unit=r"\peta\hertz")
Tab_7.addColumn(U_g, title="Grenzspannung", symbol=r"U_{g}", unit=r"\volt")
#Tab_7.show()
#Tab_7.save("Tabellen/Messwerte_Bunt.tex")
## Print Funktionen