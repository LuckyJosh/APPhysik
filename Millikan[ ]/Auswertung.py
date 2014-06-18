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

# Apparaturdaten etc.: Plattenabstand, Messstrecke, Cunningham Konstante
d, s, B = np.loadtxt("Messdaten/Apparatur.txt")
d *= 1e-03  # m
#B /=  # Skaliert mit dem Luftdruck

# Öldichte
p_oel = np.loadtxt("Messdaten/Oeldichte.txt")

# Temperatur --> Viskosität
Temp, eta = np.loadtxt("Messdaten/Viskositaet.txt", unpack=True)

# Array Viskosität[Temperatur]
Eta = dict(zip(Temp, eta))


# Gleichung für den Tröpfchenradius
def radius(eta, v_ab, v_auf):
    return np.sqrt((9 * eta * (v_ab - v_auf))/(2 * const.g * p_oel))

# Gleichung für die Tröpfchenladung
def ladung(eta, v_ab, v_auf, E):
    return (3 * const.pi * eta *
            np.sqrt((9 * eta * (v_ab - v_auf))/(4 * const.g * p_oel)) *
            (v_ab + v_auf)/E)

# Cunningham Viskosität
def korr_viskositaet(eta, r):
    return ( eta * (1/(1 + B/((const.atm/const.torr) * r))))

# Cunningham Ladung
def korr_ladung(q,r):
    return (q * (np.sqrt(1 + B/((const.atm/const.torr) * r)))**3)

def func_gerade(x,b):
    return 0*x + b
#==============================================================================
class Messwerte_298V:
    pass
#==============================================================================
# Spannung der Messwerte
U_298V = 298.0  # V

# Messwerte 298V: Zeiten, Termowiderstand(nicht gebraucht), Temperaturen
t1_ab_298V, t1_auf_298V, t2_ab_298V, t2_auf_298V, R_th_298V, T_298V = np.loadtxt("Messdaten/Messwerte_298V.txt", unpack=True)

# Mittelwerte der Zeiten
t_ab_298V = [(t1 + t2)/2 if t1 > 0 and t2 > 0 else t1 if t1 > t2 else t2
             for t1,t2 in zip(t1_ab_298V, t2_ab_298V)]
t_auf_298V = [(t1 + t2)/2 if t1 > 0 and t2 > 0 else t1 if t1 > t2 else t2
              for t1,t2 in zip(t1_auf_298V, t2_auf_298V)]

# Berechnung der Geschwindigkeiten
v_ab_298V = s/ t_ab_298V
v_ab_298V *= 1e-03
v_auf_298V = s/ t_auf_298V
v_auf_298V *= 1e-03

# Viskosität der Luft bei Messung
eta_298V = np.array([Eta[T]  for T in T_298V])
eta_298V *= 1e-05    # [Ns/m^2]


# Berechnung des EFeldes bei der Messung
E_298V = U_298V / d

# Berechnung des Radius
#print(eta_298V)
#print(v_ab_298V)
#print(v_auf_298V)
#print(v_ab_298V - v_auf_298V)

R_298V = np.zeros(len(eta_298V))
for i in range(len(eta_298V)):
    if (v_ab_298V[i] - v_auf_298V[i]) > 0:
        R_298V[i] = radius(eta_298V[i], v_ab_298V[i], v_auf_298V[i])
    else:
        R_298V[i] = 0

print("Radius 1:", R_298V)


# Korrigierte Viskosität
eta_korr_298V = np.zeros(len(eta_298V))
for i in range(len(eta_298V)):
    if R_298V[i] > 0:
        eta_korr_298V[i] = korr_viskositaet(eta_298V[i], R_298V[i])
    else:
        eta_korr_298V[i] = 0
print("Korr. Viskosität1:",eta_korr_298V)


# Berechnete Ladung
q_298V = ladung(eta_korr_298V, v_ab_298V, v_auf_298V, E_298V)
print("Ladung 1", q_298V)

# Korrigierte Ladung
q_korr_298V = np.zeros(len(q_298V))
for i in range(len(q_298V)):
    if R_298V[i] > 0:
        q_korr_298V[i] = korr_ladung(q_298V[i], R_298V[i])
    else:
        q_korr_298V[i] = 0
print("Ladung 1 korregiert", q_korr_298V)
#==============================================================================
class Messwerte_297V:
    pass
#==============================================================================
# Spannung der Messwerte
U_297V = 297.0  # V

# Messwerte 297V:  Zeiten, Termowiderstand(nicht gebraucht), Temperaturen
t1_ab_297V, t1_auf_297V, t2_ab_297V, t2_auf_297V, R_th_297V, T_297V = np.loadtxt("Messdaten/Messwerte_297V.txt", unpack=True)

# Mittelwerte der Zeiten
t_ab_297V = [(t1 + t2)/2 if t1 > 0 and t2 > 0 else t1 if t1 > t2 else t2
             for t1,t2 in zip(t1_ab_297V, t2_ab_297V)]
t_auf_297V = [(t1 + t2)/2 if t1 > 0 and t2 > 0 else t1 if t1 > t2 else t2
              for t1,t2 in zip(t1_auf_297V, t2_auf_297V)]

# Berechnung der Geschwindigkeiten
v_ab_297V = s/ t_ab_297V
v_ab_297V *= 1e-03
v_auf_297V = s/ t_auf_297V
v_auf_297V *= 1e-03

# Viskosität der Luft bei Messung
eta_297V = np.array([Eta[T]  for T in T_297V])
eta_297V *= 1e-05    # [Ns/m^2]


# Berechnung des EFeldes bei der Messung
E_297V = U_297V / d

# Berechnung des Radius
R_297V = np.zeros(len(eta_297V))
for i in range(len(eta_297V)):
    if (v_ab_297V[i] - v_auf_297V[i]) > 0:
        R_297V[i] = radius(eta_297V[i], v_ab_297V[i], v_auf_297V[i])
    else:
        R_297V[i] = 0

print("Radius 2:", R_297V)


# Korrigierte Viskosität
eta_korr_297V = np.zeros(len(eta_297V))
for i in range(len(eta_297V)):
    if R_297V[i] > 0:
        eta_korr_297V[i] = korr_viskositaet(eta_297V[i], R_297V[i])
    else:
        eta_korr_297V[i] = 0
print("Korr. Viskosität2:",eta_korr_297V)


# Berechnete Ladung
q_297V = ladung(eta_korr_297V, v_ab_297V, v_auf_297V, E_297V)
print("Ladung 2", q_297V)

# Korrigierte Ladung
q_korr_297V = np.zeros(len(q_297V))
for i in range(len(q_297V)):
    if R_297V[i] > 0:
        q_korr_297V[i] = korr_ladung(q_297V[i], R_297V[i])
    else:
        q_korr_297V[i] = 0
print("Ladung 2 korregiert", q_korr_297V)

#==============================================================================
class Messwerte_201V:
    pass
#==============================================================================
# Spannung der Messwerte
U_201V = 201.0  # V

# Messwerte 201V: Zeiten, Termowiderstand(nicht gebraucht), Temperaturen
t1_ab_201V, t1_auf_201V, t2_ab_201V, t2_auf_201V, R_th_201V, T_201V = np.loadtxt("Messdaten/Messwerte_201V.txt", unpack=True)

# Mittelwerte der Zeiten
t_ab_201V = [(t1 + t2)/2 if t1 > 0 and t2 > 0 else t1 if t1 > t2 else t2
             for t1,t2 in zip(t1_ab_201V, t2_ab_201V)]
t_auf_201V = [(t1 + t2)/2 if t1 > 0 and t2 > 0 else t1 if t1 > t2 else t2
              for t1,t2 in zip(t1_auf_201V, t2_auf_201V)]

# Berechnung der Geschwindigkeiten
v_ab_201V = s/ t_ab_201V
v_ab_201V *= 1e-03
v_auf_201V = s/ t_auf_201V
v_auf_201V *= 1e-03


# Viskosität der Luft bei Messung
eta_201V = np.array([Eta[T] for T in T_201V])
eta_201V *= 1e-05    # [Ns/m^2]



# Berechnung des EFeldes bei der Messung
E_201V = U_201V / d


R_201V = np.zeros(len(eta_201V))
for i in range(len(eta_201V)):
    if (v_ab_201V[i] - v_auf_201V[i]) > 0:
        R_201V[i] = radius(eta_201V[i], v_ab_201V[i], v_auf_201V[i])
    else:
        R_201V[i] = 0

print("Radius 3:", R_201V)


# Korrigierte Viskosität
eta_korr_201V = np.zeros(len(eta_201V))
for i in range(len(eta_201V)):
    if R_201V[i] > 0:
        eta_korr_201V[i] = korr_viskositaet(eta_201V[i], R_201V[i])
    else:
        eta_korr_201V[i] = 0
print("Korr. Viskosität2:",eta_korr_201V)


# Berechnete Ladung
q_201V = ladung(eta_korr_201V, v_ab_201V, v_auf_201V, E_201V)
print("Ladung 3", q_201V)

# Korrigierte Ladung
q_korr_201V = np.zeros(len(q_201V))
for i in range(len(q_201V)):
    if R_201V[i] > 0:
        q_korr_201V[i] = korr_ladung(q_201V[i], R_201V[i])
    else:
        q_korr_201V[i] = 0
print("Ladung 3 korregiert", q_korr_201V)





#==============================================================================
class Plots:
    pass
#==============================================================================




# Ladungen
q = np.array([])
q = np.concatenate((q, q_korr_298V))
q = np.concatenate((q, q_korr_297V))
q = np.concatenate((q, q_korr_201V))
q = np.delete(q, (np.where(q == 0))[0])
print("Ladungen",q)

# Fit 1

# Messungs Nummern
N_1 = np.array([1, 3, 6, 7, 8, 11, 12, 14])

# Fit der geringsten Ladungen
q_1 = np.array([q[0], q[2], q[5], q[6], q[7], q[10], q[11], q[13]])

popt_1, pcov_1 = curve_fit(func_gerade, N_1, q_1)
print(popt_1[0])

# Fit 2

# Messungs Nummern
N_2 = np.array([2, 4, 5, 9, 13])

# Fit der geringsten Ladungen
q_2 = np.array([q[1], q[3], q[4], q[8], q[12]])

popt_2, pcov_2 = curve_fit(func_gerade, N_2, q_2)
print(popt_2[0])


# 'Fit' 3
print(q[9])






# Messungs Nummern
N = np.arange(1, len(q)+1)
n = np.linspace(-1, 16, 20)


#Plot der Messwerte
plt.plot(N, q, "rx", label="Messwerte")

# Plot der ersten Fit Gerade
plt.plot(n, func_gerade(n, popt_1[0]), color="gray", label="Regressionsgeraden")

# Plot der zweiten Fit Gerade
plt.plot(n, func_gerade(n, popt_2[0]), color="gray")

# Plot dritter Wert
plt.plot(n, func_gerade(n, q[9]), color="gray")

plt.grid()
plt.xlabel(r"Messung  $N$")
plt.ylabel(r"Ladung  $q\ [10^{-19} \mathrm{C}]$")
plt.xlim(0,15)
plt.ylim(0,9e-19)
plt.gca().yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x,_: float(x)*1e19))


plt.legend(loc="best")
plt.savefig("Grafiken/Messwerte.pdf")
plt.show()




