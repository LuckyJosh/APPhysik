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
class Tabellen:
    pass
#==============================================================================
# Spannungen
U1 = np.ones(len(T_298V)) * U_298V
U2 = np.ones(len(T_297V)) * U_297V
U3 = np.ones(len(T_201V)) * U_201V
U = np.concatenate([U1, U2, U3])
R = np.concatenate([R_th_298V, R_th_297V, R_th_201V])


t1auf = np.concatenate([t1_auf_298V,t1_auf_297V,t1_auf_201V])
t2auf = np.concatenate([t2_auf_298V, t2_auf_297V, t2_auf_201V])
t1ab = np.concatenate([t1_ab_298V,t1_ab_297V,t1_ab_201V])
t2ab = np.concatenate([t2_ab_298V, t2_ab_297V, t2_ab_201V])
tauf = np.concatenate([t_auf_298V, t_auf_297V, t_auf_201V])
tab = np.concatenate([t_ab_298V, t_ab_297V, t_ab_201V])


Tab_Messwerte = Table(siunitx=True)
Tab_Messwerte.layout(seperator="column", title_row_seperator="double",
                     border=True)
Tab_Messwerte.label("Auswertung_Messwerte")
Tab_Messwerte.caption("Die aufgenommenen Messwerte für die Auf- und Abwärtsgeschwindigkeiten der Öltröpfchen, "+
                      "deren Mittelwert und der Wert des Thermowiderstands während der jeweiligen Messung")
Tab_Messwerte.addColumn([int(u) for u in U], title="Spannung", symbol="U", unit=r"\volt")
Tab_Messwerte.addColumn(t1auf, title="Steigzeit 1", symbol="t_{1,\\text{auf}}", unit=r"\second")
Tab_Messwerte.addColumn(t2auf, title="Steigzeit 2", symbol="t_{2,\\text{auf}}", unit=r"\second")
Tab_Messwerte.addColumn(t1ab, title="Fallzeit 1", symbol="t_{1,\\text{ab}}", unit=r"\second")
Tab_Messwerte.addColumn(t2ab, title="Fallzeit 2", symbol="t_{2,\\text{ab}}", unit=r"\second")
Tab_Messwerte.addColumn(tauf, title="Steigzeit Mittel", symbol="\overline{t_{\\text{auf}}}", unit=r"\second")
Tab_Messwerte.addColumn(tab, title="Fallzeit Mittel", symbol="\overline{t_{\\text{ab}}}", unit=r"\second")
Tab_Messwerte.addColumn(R, title="Thermistor", symbol="R", unit=r"\mega\ohm")
#Tab_Messwerte.save("Tabellen/Messwerte.tex")
#Tab_Messwerte.show()



# Ergebnisse
vauf = np.concatenate([v_auf_298V, v_auf_297V, v_auf_201V])
vab = np.concatenate([v_ab_298V, v_ab_297V, v_ab_201V])
T = np.concatenate([T_298V, T_297V, T_201V])
ETA = np.concatenate([eta_298V, eta_297V, eta_201V])


Tab_Ergebnisse = Table(siunitx=True)
Tab_Ergebnisse.layout(seperator="column", title_row_seperator="double",
                     border=True)
Tab_Ergebnisse.label("Auswertung_Ergebnisse")
Tab_Ergebnisse.caption("Aus den Messwerten berechnete Steig- und Fallgeschwindigkeiten,"+
                      "sowie die Temperatur und unkorrigierte sowie korrigierte Viskosität der Luft")
Tab_Ergebnisse.addColumn(vauf*1e03, title="Steiggeschwindigkeit", symbol="v_{\\text{auf}}", unit=r"\milli\meter\per\second")
Tab_Ergebnisse.addColumn(vab*1e03, title="Fallgeschwindigkeit", symbol="v_{\\text{ab}}", unit=r"\milli\meter\per\second")
Tab_Ergebnisse.addColumn(vab*1e03 - vauf*1e03, title="Differenzgeschwindigkeit", symbol="v_{\\text{ab}} - v_{\\text{auf}}", unit=r"\milli\meter\per\second")
Tab_Ergebnisse.addColumn([int(t) for t in T], title="Lufttemperatur", symbol="T", unit=r"\celsius")
Tab_Ergebnisse.addColumn(ETA*1e06, title="Luftviskosität", symbol="\eta_{L}", unit=r"\micro\newton\second\per\square\meter")
#Tab_Ergebnisse.addColumn(ETA_korr*1e06, title="korrigierte Luftviskosität", symbol="\eta_{L, eff}", unit=r"\micro\newton\second\per\square\meter")
#Tab_Ergebnisse.save("Tabellen/Ergebnisse.tex")
#Tab_Ergebnisse.show()

# Ergebnisse2
r = np.concatenate([R_298V, R_297V, R_201V])
r = np.delete(r, np.where(r == 0)[0])
Q = np.concatenate([q_korr_298V, q_korr_297V, q_korr_201V])
Q = np.delete(Q, np.where(Q == 0)[0])
ETA_korr = np.concatenate([eta_korr_298V, eta_korr_297V, eta_korr_201V])
ETA_korr = np.delete(ETA_korr, np.where(ETA_korr == 0)[0])

Tab_Ergebnisse2 = Table(siunitx=True)
Tab_Ergebnisse2.layout(seperator="column", title_row_seperator="double",
                     border=True)
Tab_Ergebnisse2.label("Auswertung_Ergebnisse")
Tab_Ergebnisse2.caption("Aus den brauchbaren Messwerten berechnete Radien und Ladungen der Tröpfchen,"+
                      "sowie die korrigierte Viskosität der Luft")
Tab_Ergebnisse2.addColumn(r*1e06, title="Radius", symbol="r", unit=r"\micro\meter")
Tab_Ergebnisse2.addColumn(ETA_korr*1e06, title="korrigierte Viskosität", symbol="\\eta_{\\text{L,eff}}", unit=r"\micro\newton\second\per\square\meter")
Tab_Ergebnisse2.addColumn(Q*1e19, title="Ladung", symbol="q", unit=r"\coulomb")

#Tab_Ergebnisse2.save("Tabellen/Ergebnisse2.tex")
#Tab_Ergebnisse2.show()

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
fig, ax1 = plt.subplots()
#plt.plot(N, q, "rx", label="Ladungen")
ax1.plot(-1,-1 , "kx", label="Ladungen")
ax1.plot(N_1, q_1, "rx")
ax1.plot(N_2, q_2, "gx")
ax1.plot(10, q[9], "bx")



# Plot der ersten Fit Gerade
ax1.plot(n, func_gerade(n, popt_1[0]), color="gray", label="Regressionsgeraden")

# Plot der zweiten Fit Gerade
ax1.plot(n, func_gerade(n, popt_2[0]), color="gray")

# Plot dritter Wert
ax1.plot(n, func_gerade(n, q[9]), color="gray")


# Plot der Literaturwerte
e = const.elementary_charge * 1e19
ax2 = ax1.twinx()
ax2.set_ylim(0,9)
ax2.set_yticks((1*e, 2*e, 3*e, 4*e, 5*e))
ax2.set_yticklabels(("1$e_{0}$","2$e_{0}$","3$e_{0}$","4$e_{0}$", "5$e_{0}$"))

ax1.plot(n, func_gerade(n, const.elementary_charge), "k--", label="Vielfache der Elementarladung")
ax1.plot(n, func_gerade(n, 2*const.elementary_charge), "k--")
ax1.plot(n, func_gerade(n, 3*const.elementary_charge), "k--")
ax1.plot(n, func_gerade(n, 4*const.elementary_charge), "k--")


ax1.grid()
ax1.set_xlabel(r"Messung  $N$")
ax1.set_ylabel(r"Ladung  $q\ [10^{-19} \mathrm{C}]$")
ax1.set_xlim(0,15)
ax1.set_ylim(0,9e-19)
ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x,_: float(x)*1e19))

ax1.legend(loc="best")
fig.savefig("Grafiken/Messwerte.pdf")
fig.show()



# Tabelle
e_0 = np.array([popt_1[0], popt_2[0], q[9]])
n = np.array([1,2,4])
de_0 = np.zeros(3)

for i in range(3):
    de_0[i] = np.abs((const.elementary_charge - (e_0[i]/n[i]))*100/const.elementary_charge)
print(de_0)

# Mittelwertberechnung
E_0 = np.array([e_0[i]/n[i] for i in range(3)])
E_0_err = ufloat(np.mean(E_0), np.std(E_0)/np.sqrt(len(E_0)))
print("Elementar Ladungen", E_0)
print("Mittelwert", np.mean(E_0))
print("Mittelwert Abweichung", np.std(E_0)/np.sqrt(len(E_0)))
print("Mittelwert Abweichung lit", np.abs((const.elementary_charge - np.mean(E_0))/const.elementary_charge) )




Tab_Ladung = Table(siunitx=True)
Tab_Ladung.layout(seperator="column", title_row_seperator="double",
                     border=True)
Tab_Ladung.label("Auswertung_Ladung")
Tab_Ladung.caption("Abweichung der berechneten Vielfachen der Elementarladung vom Literaturwert")
Tab_Ladung.addColumn(e_0*1e19, title="Ladung", symbol=r"n\cdot e_{0}", unit=r"\coulomb")
Tab_Ladung.addColumn(n, title="Faktor", symbol="n")
Tab_Ladung.addColumn(de_0, title="Abweichung", symbol=r"1-\frac{e_0}{e_{0,\text{lit}}}", unit=r"\percent")

#Tab_Ladung.save("Tabellen/Ladung.tex")
#Tab_Ladung.show(quiet=False)


#Avogadrokonstante
N_A = const.physical_constants["Faraday constant"][0]/E_0_err
print(N_A)
print("Abweichung Lit",np.abs((const.Avogadro - N_A)/const.Avogadro) )
