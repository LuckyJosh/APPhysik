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

def dist(x,y):
    return np.abs(x - y)


#==============================================================================
class Messung_I():
    pass
#==============================================================================
def gerade(x, m, b):
    return m * x + b

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

# Plot der Geraden durch b und g
X = np.linspace(0, 70, 1000)


for i in range(len(g_I_err)):
    plt.plot(X, gerade(X, *geradenParameter(noms(g_I_err[i]), noms(b_I_err[i]))))
plt.plot(0, 0, label="Messreihen", color="gray")

plt.xlim(0,70)
plt.ylim(0,20)
plt.legend(loc="best")

## Zoomviereck
ax = plt.axes([0.21, .465, 0.05, 0.05], axisbg="w")
ax.patch.set_alpha(0.0)
plt.setp(ax, xticks=[], yticks=[])


## ZoomPlot im Plot
x = np.linspace(5, 15, 1000)
ax = plt.axes([0.3, .5, 0.35, 0.35])
for i in range(len(g_I_err)):
    plt.plot(X, gerade(X, *geradenParameter(noms(g_I_err[i]), noms(b_I_err[i]))))
plt.setp(ax, xticks=[5, 10, 15], yticks=[5, 10, 15], xlim=[5,15], ylim=[5,15])


#plt.show()

plt.clf()



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

# Plot der Geraden durch b und g
X = np.linspace(0, 70, 1000)


for i in range(len(g_II_err)):
    plt.plot(X, gerade(X, *geradenParameter(noms(g_II_err[i]), noms(b_II_err[i]))))
plt.plot(0, 0, label="Messreihen", color="gray")
#plt.plot(X, X, "--", label="Winkelhalbierende")
plt.xlim(0,70)
plt.ylim(0,20)
plt.legend(loc="best")

## Zoomviereck
ax = plt.axes([0.175, .365, 0.07, 0.07], axisbg="w")
ax.patch.set_alpha(0.0)
plt.setp(ax, xticks=[], yticks=[])


## ZoomPlot im Plot
x = np.linspace(5, 15, 1000)
ax = plt.axes([0.3, .5, 0.35, 0.35])
for i in range(len(g_I_err)):
    plt.plot(X, gerade(X, *geradenParameter(noms(g_II_err[i]), noms(b_II_err[i]))))
plt.setp(ax, xticks=[5, 7.5, 10], yticks=[5, 7.5, 10], xlim=[5,10], ylim=[5,10])


plt.show()
plt.clf()



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


#==============================================================================
class Messung_VI():
    pass
#==============================================================================

#Laden der Daten
b_x_VI, A_x_VI, B = np.loadtxt("Messdaten/BesselRot.txt", unpack=True)

# Fehlerbehaftete Größe
b_x_VI_err = unp.uarray(b_x_VI, [l_err]*len(b_x_VI))
A_x_VI_err = unp.uarray(A_x_VI, [l_err]*len(A_x_VI))
B_err = unp.uarray(B, [l_err]*len(B))

# Abbildungsmaßstab
V_err = G_err/B_err

# gestrichene Bildweite
b_VI_err = dist(b_x_VI_err, A_x_VI_err)

# gestrichene Gegenstandsweite
g_VI_err = dist(L_x_err, A_x_VI_err)


## Print Funktionen