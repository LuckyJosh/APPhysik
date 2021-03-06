# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 23:07:29 2013

@author: Josh
"""

from __future__ import (print_function,
                        division,
                        unicode_literals,
                        absolute_import)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit
from scipy.stats import linregress
import sympy as sp
from uncertainties import ufloat
import uncertainties.unumpy as unp


def f(x, m, b):
    return m*x + b


#%%
## Bestimmung der Winkelrichtgröße

deg, F = np.loadtxt("Messwerte/Bestimmung_Winkelrichtgroesse.txt",
                    unpack=True)

F *= 1e-03  # N
rad = np.deg2rad(deg)
r, r_err = np.loadtxt("Messwerte/Bestimmung_Winkelrichtgroesse_Radius.txt",
                      unpack=True)
r *= 1e-02  # m
r_err *= 1e-02  # m

#Fehlerbehafteter Radius
ua = ufloat(r, r_err)  # m
#Winkelrichtgröße
D = (F * ua)/rad  # Nm

#Mittelwert
D_avr = np.mean(D)

#Standardabweichung des Mittelwertes
D_err = np.std(unp.nominal_values(D))/np.sqrt(len(D))

#Fehlerbehafte Winkelrichtgröße
uD = ufloat(unp.nominal_values(D_avr), unp.std_devs(D_avr) + D_err)
print(uD)
#Speichern der Daten
np.savetxt("Ausgabe/Winkelrichtgoesse.txt", D, fmt=str("%r"))
np.savetxt("Ausgabe/Winkelrichtgoesse_Mittelwert.txt", [D_avr, D_err, uD],
           fmt=str("%r"),
           header="Mittel mit Messunsicherheit \nStandardabweichung \nBeides")

#%%
## Bestimmung des EigenTraegheitsmoments
    
    # Berechnung der Peridodendauer
T_1, T_2 = np.loadtxt("Messwerte/Bestimmung_Eigendrehmoment_Periodendauer.txt",
                      unpack=True)
dim = len(T_1)
T_1 /= 5
T_2 /= 5

T_avr = np.zeros(dim)
T_avr_err = np.zeros(dim)

for i in range(dim):
        T_avr[i] = np.mean([T_1[i], T_2[i]])
        T_avr_err[i] = np.std([T_1[i], T_2[i]])/np.sqrt(dim)
uT_avr = unp.uarray(T_avr, T_avr_err)



np.savetxt("Ausgabe/Eigendrehmoment_Periodendauer_Mittelwert.txt", uT_avr,
           fmt=str("%r"))

    # Laden der Masse und der Radien
R = np.loadtxt("Messwerte/Bestimmung_Eigendrehmoment_Radius.txt",
               unpack=True)
R_err = np.loadtxt("Messwerte/Bestimmung_Eigendrehmoment_Radius_Fehler.txt")

R *= 1e-03  # m
R_err *= 1e-03  # m

uR = unp.uarray(R, R_err)


M, M_err = np.loadtxt("Messwerte/Bestimmung_Eigendrehmoment.txt",
                      usecols=(1, 2))
M *= 1e-03  # kg
M_err *= 1e-03  # kg

uM = ufloat(M, M_err)

    # Lineare Regression
popt, pcov = curve_fit(f, unp.nominal_values(uR)**2,
                       unp.nominal_values(uT_avr)**2,
                       sigma=unp.std_devs(uT_avr))
error = np.sqrt(np.diag(pcov))

    #Fehlerbehaftete Parameter
um = ufloat(popt[0], error[0])
ub = ufloat(popt[1], error[1])


I_D = (uD * ub) / (4 * const.pi**2)  # kg * m²
print(I_D.error_components().items())
print("Trägheitsmoment", I_D)
np.savetxt("Test.txt", [I_D], fmt=str("%r"))

#%%
    # Plot a² -> T²
plt.grid()
plt.ylabel("$T^{2} [\mathrm{s^{2}}]$")
plt.xlabel("$R^{2} [\mathrm{m^{2}}]$")
plt.xlim(0, 0.05)

x = np.linspace(0, max(unp.nominal_values(uR)**2)+0.01, 1000)


plt.errorbar(unp.nominal_values(uR)**2, unp.nominal_values(uT_avr)**2,
             yerr=unp.std_devs(uT_avr)**2, fmt="rx", label="Messwerte")
plt.plot(x, f(x, *popt), "b-", label="Regerssionsgrade")

plt.legend(loc="best")
plt.tight_layout()

plt.savefig("Plots/Plot_Eigendrehmoment.pdf")
    # Speichern der Regerssionsparameter m und b
np.savetxt("Ausgabe/RegerssionsParameter.txt", [um, ub], fmt=str("%r"),
           header="Steigung m \nAchsenabschnitt b")

print("X-Werte:", (uR)**2)
print("Y-Werte:", (uT_avr)**2)

#%%

## Bestimmung Tägheitsmoment Koerper
K_T_1, K_T_2, Z_T_1, Z_T_2 = np.loadtxt("Messwerte/Periodendauer_Koerper.txt",
                                        unpack=True)
K_T_1 /= 5
K_T_2 /= 5
Z_T_1 /= 5
Z_T_2 /= 5


dim = len(K_T_1)

    ## Kugel
K_T_avrs = np.zeros(dim)  # Mittelwert jeweils einer Schwingung
K_T_errs = np.zeros(dim)  # Fehler des Mittelwertes jeweils einer Schwingung
for i in range(dim):
    K_T_avrs[i] = np.mean([K_T_1[i], K_T_2[i]])
#uK_T_avrs = unp.uarray(K_T_avrs, K_T_errs)  # Fehler behafteter MW je Schwingung
K_T_avr = np.mean(K_T_avrs)
K_T_err = np.std(K_T_avrs)/(len(K_T_avrs) - 1)
uK_T_avr = ufloat(K_T_avr, K_T_err)
print("--Mittelwert Kugel", uK_T_avr)
I_K = (uD * uK_T_avr**2) / (const.pi**2 * 4)   # kgm²

#I_K -= I_D
print("--Bestimmt I_K:", I_K*1e03) 
print("Mittelwerte Kugel", K_T_avrs)


    ## Zylinder
Z_T_avrs = np.zeros(dim)
Z_T_errs = np.zeros(dim)
for i in range(dim):
    Z_T_avrs[i] = np.mean([Z_T_1[i], Z_T_2[i]])
#uZ_T_avrs = unp.uarray(Z_T_avrs, Z_T_errs)
Z_T_avr  = np.mean(Z_T_avrs)
Z_T_err  = np.std(Z_T_avrs)/(len(Z_T_avrs) - 1 )
uZ_T_avr  = ufloat(Z_T_avr, Z_T_err)

print("--Mittelwert Zylinder", uZ_T_avr)
I_Z = (uD * uZ_T_avr**2) / (4 * const.pi**2)  # kgm²

#I_Z -= I_D
print("--Bestimmtes I_Z:", I_Z*1e03)
print("Mittelwerte Zylider", Z_T_avrs)

#%%
## Berechnung Traegheitsmomente Koerper    
    ## Kugel
M_K, M_K_err, U_K, U_K_err= np.loadtxt("Messwerte/Dimension_Koerper_Kugel.txt",
                                       unpack=True)    
M_K *= 1e-03
M_K_err *= 1e-03
uM_K = ufloat(M_K, M_K_err)
print("Masse Kugel", uM_K)
U_K *= 1e-02
U_K_err *= 1e-02
uU_K = ufloat(U_K, U_K_err)

uR_K = uU_K / (2 * const.pi)
print("Radius Kugel", uR_K)

uI_K = 0.4 * uM_K * uR_K**2
print("--Trägheitsmoment Kugel", uI_K)

    ## Zylinder
M_Z, M_Z_err, D_Z, D_Z_err = np.loadtxt("Messwerte/Dimension_Koerper_Zylinder.txt",
                                        usecols=(0, 1, 2, 3), unpack=True)

M_Z *= 1e-03
M_Z_err *= 1e-03
uM_Z = ufloat(M_Z, M_Z_err)

D_Z *= 1e-02
D_Z_err *= 1e-02
uD_Z = ufloat(D_Z, D_Z_err)

uR_Z = uD_Z / 2
uI_Z = 0.5 * uM_Z * uR_Z**2
print("--Trägheitsmoment Zylinder", uI_Z)
print("Masse Zylinder", uM_Z)
print("Radius Zylinder", uR_Z)
#%%


## Speichern der Tägheitsmomente
np.savetxt("Ausgabe/Traegheitsmoment_Koerper_Kugel.txt", [I_K, uI_K],
           fmt=str("%r"), header="Gemessen [kgm²] \nTheorie [kgm²]")
np.savetxt("Ausgabe/Traegheitsmoment_Koerper_Zylinder.txt", [I_Z, uI_Z],
           fmt=str("%r"), header="Gemessen[kgm²] \nTheorie [kgm²]")


#%%
## Bestimmung der Traegheitsmomente Puppe  
P1_T_1, P1_T_2, P2_T_1, P2_T_2 = np.loadtxt("Messwerte/Periodendauer_Puppe.txt",
                                            unpack=True)   
   
P1_T_1 /= 5
P1_T_2 /= 5
P2_T_1 /= 5
P2_T_2 /= 5


dim = len(P1_T_1)

    #Pose1
P1_T_avrs = np.zeros(dim)  # Mittelwert jeweils einer Schwingung
P1_T_errs = np.zeros(dim)  # Fehler des Mittelwertes jeweils einer Schwingung
for i in range(dim):
    P1_T_avrs[i] = np.mean([P1_T_1[i], P1_T_2[i]])
#uP1_T_avrs = unp.uarray(P1_T_avrs, P1_T_errs)  # Fehler behafteter MW je Schwingung


P1_T_avr = np.mean(P1_T_avrs)
P1_T_err = np.std(P1_T_avrs)/(len(P1_T_avrs) -1)
uP1_T_avr = ufloat(P1_T_avr, P1_T_err)

print("MittelwertP1", uP1_T_avr)
I_P1 = (uD * uP1_T_avr**2) / (4 * const.pi**2)  # kgm²
#I_P1 -= I_D
print("Pose1:", I_P1)
print("MittelwerteP1", P1_T_avrs)
    #Pose2
P2_T_avrs = np.zeros(dim)  # Mittelwert jeweils einer Schwingung
P2_T_errs = np.zeros(dim)  # Fehler des Mittelwertes jeweils einer Schwingung
for i in range(dim):
    P2_T_avrs[i] = np.mean([P2_T_1[i], P2_T_2[i]])
#uP2_T_avrs = unp.uarray(P2_T_avrs, P2_T_errs)  # Fehler behafteter MW je Schwingung
P2_T_avr = np.mean(P2_T_avrs)
P2_T_err = np.std(P2_T_avrs)/(len(P2_T_avrs) -1)
uP2_T_avr = ufloat(P2_T_avr, P2_T_err)

print("MittelwertP2", uP2_T_avr)
I_P2 = (uD * uP2_T_avr**2) / (4 * const.pi**2)  # kgm²
#I_P2 -= I_D
print("Pose2:", I_P2)
print("MittelwerteP2", P2_T_avrs)

#%%
## Berechnung Traegheitsmomente Puppenteile

D_a, D_b, D_t, D_k = np.loadtxt("Messwerte/Dimension_Puppe_Durchmesser.txt",
                                unpack=True)
D_a *= 1e-03  # m
D_b *= 1e-03  # m
D_t *= 1e-03  # m
D_k *= 1e-03  # m

L_a, L_b, L_t, L_k = np.loadtxt("Messwerte/Dimension_Puppe_Laengen.txt",
                                unpack=True)

L_a *= 1e-03  # m
L_b *= 1e-03  # m
L_t *= 1e-03  # m
L_k *= 1e-03  # m

err_1, err_2 = np.loadtxt("Messwerte/Dimension_Puppe_Fehler.txt",
                          unpack=True)
err_1 *= 1e-03  # m
err_2 *= 1e-03  # m

M_P, M_P_err = np.loadtxt("Messwerte/Dimension_Puppe_Masse.txt",
                          unpack=True)
M_P *= 1e-03
M_P_err *= 1e-03

uM_P = ufloat(M_P, M_P_err)


# Fehlerbehaftete Messgroeßen
D_a_avr = np.mean(D_a)
D_b_avr = np.mean(D_b)
D_t_avr = np.mean(D_t)
D_k_avr = np.mean(D_k)

D_a_avr_err = np.std(D_a)/np.sqrt(len(D_a))
D_b_avr_err = np.std(D_b)/np.sqrt(len(D_b))
D_t_avr_err = np.std(D_t)/np.sqrt(len(D_t))
D_k_avr_err = np.std(D_k)/np.sqrt(len(D_k))

uD_a = ufloat(D_a_avr, D_a_avr_err)
uD_b = ufloat(D_b_avr, D_b_avr_err)
uD_t = ufloat(D_t_avr, D_t_avr_err)
uD_k = ufloat(D_k_avr, D_k_avr_err)

uR_a = uD_a / 2  # m
uR_b = uD_b / 2  # m
uR_t = uD_t / 2  # m
uR_k = uD_k / 2  # m


uL_a = ufloat(L_a, err_1)
uL_b = ufloat(L_b, err_1)
uL_t = ufloat(L_t, err_1)
uL_k = ufloat(L_k, err_2)

print("Durchmesser", uD_k, uD_t, uD_a, uD_b)
print("Höhen", uL_k, uL_t, uL_a, uL_b)
#Volumen der Körperteile(Zylinder)


def V_Z(r, h):
    return 2 * const.pi * r * h

V_a = V_Z(uR_a, uL_a)
V_b = V_Z(uR_b, uL_b)
V_t = V_Z(uR_t, uL_t)
V_k = V_Z(uR_k, uL_k)
V_P = V_a + V_b + V_t + V_k

uM_a = V_a/V_P * uM_P
uM_b = V_b/V_P * uM_P
uM_t = V_t/V_P * uM_P
uM_k = V_k/V_P * uM_P
#print(V_a/V_P*100, V_b/V_P*100, V_t/V_P*100, V_k/V_P*100)
print("Massen", uM_k, uM_t, uM_a, uM_b)




def I_Z_v(m, r):
    return 0.5 * m * r**2


def I_Z_h(m, h, r):
    return m * ((r**2) / 4 + (h**2) / 12)


def I_Steiner(I, a, m):
    return I + (m * a**2)

# I mit vertikaler Drehachse
I_a_v = I_Z_v(uM_a, uR_a)
I_b_v = I_Z_v(uM_b, uR_b)
I_t_v = I_Z_v(uM_t, uR_t)
I_k_v = I_Z_v(uM_k, uR_k)

# I mit horizontaler Drehachse

I_a_h = I_Z_h(uM_a, uL_a, uR_a)
I_b_h = I_Z_h(uM_b, uL_b, uR_b)

# Verschiebungen der Rotationsachse
P1_a = uL_a/2 + uR_t
P2_a = uL_a/2 + uR_t

P1_b = uR_t - uR_b
P2_b = uL_b/2 + uR_t

# Gesamt I Pose1

I_ges_P1 = (I_t_v + I_k_v + 2 * I_Steiner(I_a_h, P1_a, uM_a) +
            2 * I_Steiner(I_b_v, P1_b, uM_b))

# Gesamt I Pose2
I_ges_P2 = (I_t_v + I_k_v + 2 * I_Steiner(I_a_h, P2_a, uM_a) +
            2 * I_Steiner(I_b_h, P2_b, uM_b))

I_P2_test = unp.uarray(np.zeros(3), np.zeros(3))
I_P2_test[0] = I_t_v + I_k_v 
I_P2_test[1] = 2* (I_a_h + (uM_a * P2_a**2)) 
I_P2_test[2] = 2* (I_b_h + (uM_b * P2_b**2))

print("TEST-1:", P2_b, "=", uL_b/2, "+", uR_t)
print("TEST", I_P2_test)
print("Vertikale Momente", I_k_v, I_t_v, I_a_v, I_b_v)
print("Horizontale Momente", I_a_h, I_b_h)
print(I_ges_P1, I_ges_P2)
#%%

# Speichern der Traegheitmomente
np.savetxt("Ausgabe/Trageheitsmoment_Puppe_Pose1.txt",
           [I_P1, I_ges_P1], fmt=str("%r"),
           header="Gemessen [kgm²] \nBerechnet [kgm²]")
np.savetxt("Ausgabe/Trageheitsmoment_Puppe_Pose2.txt",
           [I_P2, I_ges_P2], fmt=str("%r"),
           header="Gemessen [kgm²] \nBerechnet [kgm²]")

## Relative Fehler
r_K = (uI_K.n - I_Z.n)/uI_K.n
r_Z = (uI_Z.n - I_Z.n)/uI_Z.n
r_P1 = (I_ges_P1.n - I_P1.n)/I_ges_P1.n
r_P2 = (I_ges_P2.n - I_P2.n)/I_ges_P2.n


print("realtive Unterschiede:", r_K, r_Z, r_P1, r_P2)
print("mittel der Unteschiede:",np.mean([r_K, r_Z, r_P1, r_P2]))



