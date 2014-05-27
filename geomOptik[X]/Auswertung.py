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
import sympy as sp
import uncertainties as unc
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)



from aputils.utils import Quantity, ErrorEquation
from aputils.latextables.tables import Table


def dist(x, y):
    return np.abs(x - y)


def gerade(x, m, b):
    return m * x + b

def schnittpunkt(xy1, xy2):
    pass


umean = unc.wrap(np.mean)

#==============================================================================
class Messung_I():
    pass
#==============================================================================
def geradenParameter(x,y):
    m = -y/x
    b = y
    return m, b

# Ladne der Daten
L_x, G = np.loadtxt("Messdaten/Gegenstand.txt", unpack=True)
b_x_I, l_x_I = np.loadtxt("Messdaten/BekannteLinse.txt", unpack=True)

# Ladens des Fehlers
l_err = np.loadtxt("Messdaten/Fehler.txt")

# Fehlerbehaftete Größe
L_x_err = ufloat(L_x, l_err)
G_err = ufloat(G, l_err)
b_x_I_err = unp.uarray(b_x_I, [l_err]*len(b_x_I))
l_x_I_err = unp.uarray(l_x_I, [l_err]*len(l_x_I))

# Bildweite
b_I_err = dist(b_x_I_err, l_x_I_err)


# Gegenstandsweite
g_I_err = dist(L_x_err, l_x_I_err)

print("Bild-& GegenstandsweiteI:", g_I_err, b_I_err)
print("Brennweiten:\n", umean(1/(1/g_I_err + 1/b_I_err)))
print("Abbildungsgesetz:", g_I_err[1]/b_I_err[1], g_I_err[4]/b_I_err[4])
print("Abbildungsgesetz:", ufloat(3, 0.1)/ufloat(2, 0.1),
      ufloat(3, 0.1)/ufloat(1, 0.1))
# Plot der Geraden durch b und g
X = np.linspace(0, 70, 1000)


for i in range(len(g_I_err)):
    plt.plot(X, gerade(X, *geradenParameter(noms(g_I_err[i]), noms(b_I_err[i]))))
plt.plot(0, 0, label="Messreihen", color="gray")
plt.plot(9.81, 9.81, "o", color="black", label="Schnittpunkt")
plt.grid()
plt.xlabel("Gegenstandsweite $g \ [\mathrm{cm}]$", fontsize=14, family="serif")
plt.ylabel("Bildweite $b \ [\mathrm{cm}]$", fontsize=14, family="serif")
plt.xlim(0,70)
plt.ylim(0,20)
plt.legend(loc="best")

## Zoomviereck
ax = plt.axes([0.21, .465, 0.05, 0.05], axisbg="w")
ax.patch.set_alpha(0.0)
plt.setp(ax, xticks=[], yticks=[])


## ZoomPlot im Plot
x = np.linspace(5, 15, 1000)
ax = plt.axes([0.28, .5, 0.35, 0.35])
for i in range(len(g_I_err)):
    plt.plot(X, gerade(X, *geradenParameter(noms(g_I_err[i]), noms(b_I_err[i]))))
plt.setp(ax, xticks=[5, 10, 15], yticks=[5, 10, 15], xlim=[5,15], ylim=[5,15])
plt.plot(X,X, "--", color="gray")
plt.plot(9.81, 9.81, "o", color="black", label="Schnittpunkt")
ax.grid()
#plt.show()
plt.savefig("Grafiken/Messwerte_Bekannt.pdf")
plt.clf()

#Tabelle
T_I = Table(siunitx=True)
T_I.layout(seperator="column", title_row_seperator="double", border=True)
T_I.caption("Messwerte zur Überprüfung der bekannten Brennweite")
T_I.label("Auswertung_Messwerte_I")
T_I.addColumn(b_x_I_err, title="Pos. Bild",
               symbol="x_{B}", unit="\centi\meter")
T_I.addColumn(l_x_I_err, title="Pos. Linse",
               symbol="x_{L}", unit="\centi\meter")
T_I.addColumn(g_I_err, title="Gegenstandsweite",
               symbol="g", unit="\centi\meter")
T_I.addColumn(b_I_err, title="Bildweite",
               symbol="b", unit="\centi\meter")

#T_I.show()
#T_I.save("Tabellen/Messwerte_I.tex")

#==============================================================================
class Messung_II():
    pass
#==============================================================================

# Ladne der Daten
b_x_II, l_x_II = np.loadtxt("Messdaten/UnbekannteLinse .txt", unpack=True)

# Fehlerbehaftete Größe
b_x_II_err = unp.uarray(b_x_II, [l_err]*len(b_x_II))
l_x_II_err = unp.uarray(l_x_II, [l_err]*len(l_x_II))

# Bildweite
b_II_err = dist(b_x_II_err, l_x_II_err)

# Gegenstandsweite
g_II_err = dist(L_x_err, l_x_II_err)




print("Bild-& GegenstandsweiteII:", g_II_err, b_II_err)
print("Brennweiten:\n", umean(1/(1/g_II_err + 1/b_II_err)))
# Plot der Geraden durch b und g
X = np.linspace(0, 70, 1000)


for i in range(len(g_II_err)):
    plt.plot(X, gerade(X, *geradenParameter(noms(g_II_err[i]), noms(b_II_err[i]))))
plt.plot(0, 0, label="Messreihen", color="gray")
plt.plot(7.52, 7.52, "ko", label="Schnittpunkt")
plt.xlabel("Gegenstandsweite $g \ [\mathrm{cm}]$", fontsize=14, family="serif")
plt.ylabel("Bildweite $b \ [\mathrm{cm}]$", fontsize=14, family="serif")
plt.xlim(0,70)
plt.ylim(0,20)
plt.grid()
plt.legend(loc="best")

## Zoomviereck
ax = plt.axes([0.176, .37, 0.06, 0.06], axisbg="w")
ax.patch.set_alpha(0.0)
plt.setp(ax, xticks=[], yticks=[])


## ZoomPlot im Plot
x = np.linspace(5, 15, 1000)
ax = plt.axes([0.28, .5, 0.35, 0.35])
for i in range(len(g_I_err)):
    plt.plot(X, gerade(X, *geradenParameter(noms(g_II_err[i]), noms(b_II_err[i]))))
plt.setp(ax, xticks=[5, 7.5, 10], yticks=[5, 7.5, 10], xlim=[5,10], ylim=[5,10])
plt.plot(7.52, 7.52, "ko", label="Schnittpunkt")
plt.plot(X, X, "--", color="gray" )
plt.grid()
#plt.show()
plt.savefig("Grafiken/Messwerte_Unbekannt.pdf")
plt.clf()

#Tabelle
T_II = Table(siunitx=True)
T_II.layout(seperator="column", title_row_seperator="double", border=True)
T_II.caption("Messwerte zur Berechnung der unbekannten Brennweite")
T_II.label("Auswertung_Messwerte_II")
T_II.addColumn(b_x_II_err, title="Pos. Bild",
               symbol="x_{B}", unit="\centi\meter")
T_II.addColumn(l_x_II_err, title="Pos. Linse",
               symbol="x_{L}", unit="\centi\meter")
T_II.addColumn(g_II_err, title="Gegenstandsweite",
               symbol="g", unit="\centi\meter")
T_II.addColumn(b_II_err, title="Bildweite",
               symbol="b", unit="\centi\meter")

#T_II.show()
#T_II.save("Tabellen/Messwerte_II.tex")

#==============================================================================
class Messung_III():
    pass
#==============================================================================

#Laden der Daten
b_x_III, l_x_1_III, l_x_2_III = np.loadtxt("Messdaten/BesselWeiß.txt", unpack=True)

# Fehlerbehaftete Größe
b_x_III_err = unp.uarray(b_x_III, [l_err]*len(b_x_III))
l_x_1_III_err = unp.uarray(l_x_1_III, [l_err]*len(l_x_1_III))
l_x_2_III_err = unp.uarray(l_x_2_III, [l_err]*len(l_x_2_III))

# Bildweite 1
b_1_III_err = dist(b_x_III_err, l_x_1_III_err)

# Bildweite 2
b_2_III_err = dist(b_x_III_err, l_x_2_III_err)

# Gegenstandsweite 1
g_1_III_err = dist(L_x_err, l_x_1_III_err)

# Gegenstandsweite 2
g_2_III_err = dist(L_x_err, l_x_2_III_err)

# Linsenabstand d
d_III_err = dist(l_x_1_III_err, l_x_2_III_err)

#Gesamtabstand e
e_III_err = dist(L_x_err, b_x_III_err)

# Brennweite der Linse f
f_III_err = (e_III_err**2 - d_III_err**2)/(4 * e_III_err)
f_III_avr_err = umean(f_III_err)

#Ausgabe der Ergebnisse
print("Brennweiten III:\n", f_III_err, "\n Mittelwert:", f_III_avr_err)

# Tabelle
T_III = Table(siunitx=True)
T_III.layout(seperator="column", title_row_seperator="double", border=True)
T_III.caption("Messwerte und Ergebnisse nach Bessel mit weißem Licht")
T_III.label("Auswertung_Messwerte_III")
T_III.addColumn(b_x_III_err, title="Pos. Bild",
                symbol="x_{B}", unit="\centi\meter")
T_III.addColumn(l_x_1_III_err, title="Pos. Linse 1",
                symbol="x_{L,1}", unit="\centi\meter")
T_III.addColumn(l_x_2_III_err, title="Pos. Linse 2",
                symbol="x_{L,2}", unit="\centi\meter")
T_III.addColumn(d_III_err, title="Linsenabstand",
                symbol="d", unit="\centi\meter")
T_III.addColumn(e_III_err, title="Gesamtabstand",
                symbol="e", unit="\centi\meter")
T_III.addColumn(f_III_err, title="Brennweite",
                symbol="f", unit="\centi\meter")
#T_III.save("Tabellen/Messwerte_III.tex")

# Fehlergleichung
err_e, err_d = sp.var("e d")
err_func_f = (err_e**2 - err_d**2)/(4 * err_e)
err_f = ErrorEquation(err_func_f, name="f")
print(err_f.std)


#==============================================================================
class Messung_IV():
    pass
#==============================================================================

#Laden der Daten
b_x_IV, l_x_1_IV, l_x_2_IV = np.loadtxt("Messdaten/BesselBlau.txt", unpack=True)

# Fehlerbehaftete Größe
b_x_IV_err = unp.uarray(b_x_IV, [l_err]*len(b_x_IV))
l_x_1_IV_err = unp.uarray(l_x_1_IV, [l_err]*len(l_x_1_IV))
l_x_2_IV_err = unp.uarray(l_x_2_IV, [l_err]*len(l_x_2_IV))

# Bildweite 1
b_1_IV_err = dist(b_x_IV_err, l_x_1_IV_err)

# Bildweite 2
b_2_IV_err = dist(b_x_IV_err, l_x_2_IV_err)

# Gegenstandsweite 1
g_1_IV_err = dist(L_x_err, l_x_1_IV_err)

# Gegenstandsweite 2
g_2_IV_err = dist(L_x_err, l_x_2_IV_err)

# d 1
d_1_IV_err = dist(g_1_IV_err, b_1_IV_err)

# d 2
d_2_IV_err = dist(g_2_IV_err, b_2_IV_err)

# Mittelwert d
d_IV_err = (d_1_IV_err + d_2_IV_err)/2

#Gesamtabstand e
e_IV_err = dist(L_x_err, b_x_IV_err)

# Brennweite der Linse f
f_IV_err = (e_IV_err**2 - d_IV_err**2)/(4 * e_IV_err)
f_IV_avr_err = umean(f_IV_err)

#Ausgabe der Ergebnisse
print("Brennweiten IV:\n", f_IV_err, "\n Mittelwert:", f_IV_avr_err)

T_IV = Table(siunitx=True)
T_IV.layout(seperator="column", title_row_seperator="double", border=True)
T_IV.caption("Messwerte und Ergebnisse nach Bessel mit blauem Licht")
T_IV.label("Auswertung_Messwerte_III_b")
T_IV.addColumn(b_x_IV_err, title="Pos. Bild",
               symbol="x_{B}", unit="\centi\meter")
T_IV.addColumn(l_x_1_IV_err, title="Pos. Linse 1",
               symbol="x_{L,1}", unit="\centi\meter")
T_IV.addColumn(l_x_2_IV_err, title="Pos. Linse 2",
               symbol="x_{L,2}", unit="\centi\meter")
T_IV.addColumn(d_IV_err, title="Linsenabstand",
               symbol="d", unit="\centi\meter")
T_IV.addColumn(e_IV_err, title="Gesamtabstand",
               symbol="e", unit="\centi\meter")
T_IV.addColumn(f_IV_err, title="Brennweite",
               symbol="f_{b}", unit="\centi\meter")
#T_IV.save("Tabellen/Messwerte_IIIb.tex")
#==============================================================================
class Messung_V():
    pass
#==============================================================================

#Laden der Daten
b_x_V, l_x_1_V, l_x_2_V = np.loadtxt("Messdaten/BesselRot.txt", unpack=True)

# Fehlerbehaftete Größe
b_x_V_err = unp.uarray(b_x_V, [l_err]*len(b_x_V))
l_x_1_V_err = unp.uarray(l_x_1_V, [l_err]*len(l_x_1_V))
l_x_2_V_err = unp.uarray(l_x_2_V, [l_err]*len(l_x_2_V))

# Bildweite 1
b_1_V_err = dist(b_x_V_err, l_x_1_V_err)

# Bildweite 2
b_2_V_err = dist(b_x_V_err, l_x_2_V_err)

# Gegenstandsweite 1
g_1_V_err = dist(L_x_err, l_x_1_V_err)

# Gegenstandsweite 2
g_2_V_err = dist(L_x_err, l_x_2_V_err)

# d 1
d_1_V_err = dist(g_1_V_err, b_1_V_err)

# d 2
d_2_V_err = dist(g_2_V_err, b_2_V_err)

# Mittelwert d
d_V_err = (d_1_V_err + d_2_V_err)/2

#Gesamtabstand e
e_V_err = dist(L_x_err, b_x_V_err)

# Brennweite der Linse f
f_V_err = (e_V_err**2 - d_V_err**2)/(4 * e_V_err)
f_V_avr_err = umean(f_V_err)

#Ausgabe der Ergebnisse
print("Brennweiten V:\n", f_V_err, "\n Mittelwert:", f_V_avr_err)

T_V = Table(siunitx=True)
T_V.layout(seperator="column", title_row_seperator="double", border=True)
T_V.caption("Messwerte und Ergebnisse nach Bessel mit rotem Licht")
T_V.label("Auswertung_Messwerte_III_r")
T_V.addColumn(b_x_V_err, title="Pos. Bild",
               symbol="x_{B}", unit="\centi\meter")
T_V.addColumn(l_x_1_V_err, title="Pos. Linse 1",
               symbol="x_{L,1}", unit="\centi\meter")
T_V.addColumn(l_x_2_V_err, title="Pos. Linse 2",
               symbol="x_{L,2}", unit="\centi\meter")
T_V.addColumn(d_V_err, title="Linsenabstand",
               symbol="d", unit="\centi\meter")
T_V.addColumn(e_V_err, title="Gesamtabstand",
               symbol="e", unit="\centi\meter")
T_V.addColumn(f_V_err, title="Brennweite",
               symbol="f_{r}", unit="\centi\meter")
#T_V.save("Tabellen/Messwerte_IIIr.tex")
#==============================================================================
class Messung_VI():
    pass
#==============================================================================


def abbeX_g(V):
    return 1 + (1 / V)


def abbeX_b(V):
    return 1 + V



#Laden der Daten
b_x_VI, A_x_VI, B = np.loadtxt("Messdaten/Abbe.txt", unpack=True)

# Fehlerbehaftete Größe
b_x_VI_err = unp.uarray(b_x_VI, [l_err]*len(b_x_VI))
A_x_VI_err = unp.uarray(A_x_VI, [l_err]*len(A_x_VI))
B_err = unp.uarray(B, [l_err]*len(B))

# Abbildungsmaßstab
V_err = B_err/G_err

# gestrichene Bildweite
b_VI_err = dist(b_x_VI_err, A_x_VI_err)

# gestrichene Gegenstandsweite
g_VI_err = dist(L_x_err, A_x_VI_err)



T_VI = Table(siunitx=True)
T_VI.layout(seperator="column", title_row_seperator="double", border=True)
T_VI.caption("Messwerte und Ergebnisse nach der Methode von Abbe")
T_VI.label("Auswertung_Messwerte_VI")
T_VI.addColumn(b_x_VI_err, title="Pos. Bild",
               symbol="x_{B}", unit="\centi\meter")
T_VI.addColumn(A_x_VI_err, title="Pos. Referenzpunkt",
               symbol="x_{A}", unit="\centi\meter")
T_VI.addColumn(B_err, title="Bildgröße",
               symbol="B", unit="\centi\meter")
T_VI.addColumn(g_VI_err, title="Gegenstandsweite",
               symbol="g'", unit="\centi\meter")
T_VI.addColumn(b_VI_err, title="Bildweite",
               symbol="b'", unit="\centi\meter")
T_VI.addColumn(V_err, title="Abbildungsmaßstab",
               symbol="V")
#T_VI.show()
#T_VI.save("Tabellen/Messwerte_VI.tex")





#Regression der Messwerte
popt_VI_g, pcov_VI_g = curve_fit(gerade, abbeX_g(noms(V_err)), noms(g_VI_err),
                                 sigma=stds(g_VI_err))
error_VI_g = np.sqrt(np.diag(pcov_VI_g))
param_VI_g_M = ufloat(popt_VI_g[0], error_VI_g[0])
param_VI_g_B = ufloat(popt_VI_g[1], error_VI_g[1])
print("Regressionsparameter g(V):\n", param_VI_g_M, param_VI_g_B)


def geradeF(x, b):
    return noms(param_VI_g_M) * x + b

popt_VI_b, pcov_VI_b = curve_fit(gerade, abbeX_b(noms(V_err)), noms(b_VI_err),
                                 sigma=stds(b_VI_err))
error_VI_b = np.sqrt(np.diag(pcov_VI_b))
param_VI_b_M = ufloat(popt_VI_b[0], error_VI_b[0])
param_VI_b_B = ufloat(popt_VI_b[1], error_VI_b[1])
print("Regressionsparameter b(V):\n",  param_VI_b_M, param_VI_b_B)

print((param_VI_b_M + param_VI_g_M)/2)
print((ufloat(25,1) + ufloat(26,1))/2)

# Erstellen der Plots
X = np.linspace(0, 5, num=1000)
plt.grid()
plt.xlabel("$1 + V$", fontsize=14, family="serif")
plt.ylabel("Bildweite $b\prime \ [\mathrm{cm}]$")
plt.xlim(2, 3.75)
plt.ylim(60, 110)
plt.errorbar(abbeX_b(noms(V_err)), noms(b_VI_err),
             xerr=stds(abbeX_b(V_err)), yerr=stds(b_VI_err),
             fmt="rx", label="Messwerte")
plt.plot(X, gerade(X, *popt_VI_b), color="gray", label="Regressionsgerade")
plt.legend(loc="best")
#plt.show()
plt.savefig("Grafiken/Messwerte_Abbe1.pdf")
plt.clf()

plt.grid()
plt.xlabel("$1 + V^{-1}$", fontsize=14, family="serif")
plt.ylabel("Gegenstandsweite $g\prime \ [\mathrm{cm}]$")
plt.xlim(1.25, 2)
plt.ylim(20, 40)
plt.errorbar(noms(abbeX_g(V_err)), noms(g_VI_err),
             xerr=stds(abbeX_g(V_err)), yerr=stds(g_VI_err),
             fmt="rx", label="Messwerte")
plt.plot(X, gerade(X, *popt_VI_g), color="gray", label="Regressionsgerade")
plt.legend(loc="best")
#plt.show()
plt.savefig("Grafiken/Messwerte_Abbe2.pdf")



# Fehlergleichung
err_G, err_B = sp.var("G B")
err_func_V = (err_B)/(err_G)
err_V= ErrorEquation(err_func_V, name="V")
print(err_V.std)