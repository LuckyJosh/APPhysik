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


#==============================================================================
class Allgemein:
    pass
#==============================================================================

# Anzahl der Zellen, Seitenlänge, Offset der Zellen und der Lampe

N_zelle, s_zelle, off_zelle, off_lampe = np.loadtxt("Messdaten/AufbauDaten.txt",
                                                    unpack=True)

# Fehler der Messgrößen: Länge, Strom, Spannung
l_err, i_err, u_err = np.loadtxt("Messdaten/Messfehler.txt", unpack=True)

# Fehlerbehaftete Länge
l_zelle_err = ufloat(s_zelle, l_err)

# Berechnung der Solarzellenfläche
A_zelle_err = N_zelle * (l_zelle_err**2)
print("Fläche der Solarzellen:", A_zelle_err, "cm^2")

# Fehlerbehaftete Offsets
off_zelle_err = ufloat(off_zelle, l_err)
off_lampe_err = ufloat(off_lampe, l_err)

# Bestimmung des Gesamtoffsets des Abstands
l_offset_err = off_lampe_err + off_zelle_err
#print("Offset", l_offset_err)

def AbstandOhneOffset(L):
    return L - l_offset_err

#==============================================================================
class KurzschlussStrom:
    pass
#==============================================================================

# Messdaten: Abstände, Kurzschlussstrom
L_1, I_k_1 = np.loadtxt("Messdaten/KurzschlussStrom.txt", unpack=True)

# Fehlerehaftete Abstände und Ströme
L_1_err = unp.uarray(L_1, [l_err]*len(L_1))
I_k_1_err = unp.uarray(I_k_1, [i_err]*len(I_k_1))
#print("Abstände mit", L_1_err)

# Umrechnung der Messwerte
L_1_err = AbstandOhneOffset(L_1_err)
#print("Abstände ohne", L_1_err)










#==============================================================================
class LeerlaufSpannung:
    pass
#==============================================================================

# Messdaten: Abstand, Spannung
L_2, U_2 = np.loadtxt("Messdaten/Leerlaufspannung.txt", unpack=True)

# Fehlerbehaftet: Abstand, Spannung
L_2_err = unp.uarray(L_2, len(L_2)*[l_err])
U_2_err = unp.uarray(U_2, len(U_2)*[u_err])

# Umrechnung der Abstände
L_2_err = AbstandOhneOffset(L_2_err)




#==============================================================================
class Kennkurve_30mA:
    pass
#==============================================================================

# Messdaten : Widerstand, Strom, Spannung
R_1, I_1, U_1 = np.loadtxt("Messdaten/Kennkurve_30mA.txt",unpack=True)

# Fehlerbehaftet: Strom, Spannung
I_1_err = unp.uarray(-I_1, len(I_1)*[i_err])
U_1_err = unp.uarray(U_1, len(U_1)*[u_err])


# Plot der Messwerte (U/I)
plt.plot(noms(U_1_err), noms(I_1_err), "xr", label="Messwerte")


# Plot-Einstellungen
plt.grid()
plt.xlabel("Spannung $U\ [\mathrm{V}] $", family="serif", fontsize="14")
plt.ylabel("Stromstärke $I\ [\mathrm{mA}] $", family="serif", fontsize="14")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("Grafiken/Kennlinie_30mA.pdf")
plt.show()
plt.clf()


#==============================================================================
# Maximale Leistung...
#==============================================================================



# Berechnung der Leistung
P_1_err = U_1_err * I_1_err

# Berechnung des Lastwiderstands
R_last_1_err = np.abs(U_1_err / I_1_err)
#print("Lastwiderstand", R_last_1_err)

# Plot der Messwerte (U/I)
plt.plot(noms(R_last_1_err), noms(P_1_err), "xr", label="Messwerte")


# Plot-Einstellungen
plt.grid()
plt.xlabel("Widerstand $R_{last}\ [\mathrm{V}] $", family="serif", fontsize="14")
plt.ylabel("Leistung $P\ [\mathrm{mA}] $", family="serif", fontsize="14")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("Grafiken/Leistung_30mA.pdf")
plt.show()
plt.clf()


#==============================================================================
class Kennkurve_50mA:
    pass
#==============================================================================

# Messdaten : Widerstand, Strom, Spannung
R_2, I_2, U_2 = np.loadtxt("Messdaten/Kennkurve_50mA.txt",unpack=True)


# Fehlerbehaftet: Strom, Spannung
I_2_err = unp.uarray(-I_2, len(I_2)*[i_err])
U_2_err = unp.uarray(U_2, len(U_2)*[u_err])


# Plot der Messwerte (U/I)
plt.plot(noms(U_2_err), noms(I_2_err), "xr", label="Messwerte")


# Plot-Einstellungen
plt.grid()
plt.xlabel("Spannung $U\ [\mathrm{V}] $", family="serif", fontsize="14")
plt.ylabel("Stromstärke $I\ [\mathrm{mA}] $", family="serif", fontsize="14")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("Grafiken/Kennlinie_50mA.pdf")
plt.show()
plt.clf()

#==============================================================================
# Maximale Leistung...
#==============================================================================



# Berechnung der Leistung
P_2_err = U_2_err * I_2_err

# Berechnung des Lastwiderstands
R_last_2_err = np.abs(U_2_err / I_2_err)
#print("Lastwiderstand", R_last_1_err)

# Plot der Messwerte (U/I)
plt.plot(noms(R_last_2_err), noms(P_2_err), "xr", label="Messwerte")


# Plot-Einstellungen
plt.grid()
plt.xlabel("Widerstand $R_{last}\ [\mathrm{V}] $", family="serif", fontsize="14")
plt.ylabel("Leistung $P\ [\mathrm{mA}] $", family="serif", fontsize="14")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("Grafiken/Leistung_50mA.pdf")
plt.show()
plt.clf()

#==============================================================================
class Kennkurve_75mA:
    pass
#==============================================================================

# Messdaten : Widerstand, Strom, Spannung
R_3, I_3, U_3 = np.loadtxt("Messdaten/Kennkurve_75mA.txt",unpack=True)

# Fehlerbehaftet: Strom, Spannung
I_3_err = unp.uarray(-I_3, len(I_3)*[i_err])
U_3_err = unp.uarray(U_3, len(U_3)*[u_err])


# Plot der Messwerte (U/I)
plt.plot(noms(U_3_err), noms(I_3_err), "xr", label="Messwerte")


# Plot-Einstellungen
plt.grid()
plt.xlabel("Spannung $U\ [\mathrm{V}] $", family="serif", fontsize="14")
plt.ylabel("Stromstärke $I\ [\mathrm{mA}] $", family="serif", fontsize="14")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("Grafiken/Kennlinie_75mA.pdf")
plt.show()
plt.clf()

#==============================================================================
# Maximale Leistung...
#==============================================================================



# Berechnung der Leistung
P_3_err = U_3_err * I_3_err

# Berechnung des Lastwiderstands
R_last_3_err = np.abs(U_3_err / I_3_err)
#print("Lastwiderstand", R_last_1_err)

# Plot der Messwerte (U/I)
plt.plot(noms(R_last_3_err), noms(P_3_err), "xr", label="Messwerte")


# Plot-Einstellungen
plt.grid()
plt.xlabel("Widerstand $R_{last}\ [\mathrm{V}] $", family="serif", fontsize="14")
plt.ylabel("Leistung $P\ [\mathrm{mA}] $", family="serif", fontsize="14")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("Grafiken/Leistung_75mA.pdf")
plt.show()
plt.clf()
#==============================================================================
class Kennkurve_100mA:
    pass
#==============================================================================

# Messdaten : Widerstand, Strom, Spannung
R_4, I_4, U_4 = np.loadtxt("Messdaten/Kennkurve_100mA.txt",unpack=True)

# Fehlerbehaftet: Strom, Spannung
I_4_err = unp.uarray(-I_4, len(I_4)*[i_err])
U_4_err = unp.uarray(U_4, len(U_4)*[u_err])


# Plot der Messwerte (U/I)
plt.plot(noms(U_4_err), noms(I_4_err), "xr", label="Messwerte")


# Plot-Einstellungen
plt.grid()
plt.xlabel("Spannung $U\ [\mathrm{V}] $", family="serif", fontsize="14")
plt.ylabel("Stromstärke $I\ [\mathrm{mA}] $", family="serif", fontsize="14")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("Grafiken/Kennlinie_75mA.pdf")
plt.show()
plt.clf()

#==============================================================================
# Maximale Leistung...
#==============================================================================



# Berechnung der Leistung
P_4_err = U_4_err * I_4_err

# Berechnung des Lastwiderstands
R_last_4_err = np.abs(U_4_err / I_4_err)
#print("Lastwiderstand", R_last_1_err)

# Plot der Messwerte (U/I)
plt.plot(noms(R_last_4_err), noms(P_4_err), "xr", label="Messwerte")


# Plot-Einstellungen
plt.grid()
plt.xlabel("Widerstand $R_{last}\ [\mathrm{V}] $", family="serif", fontsize="14")
plt.ylabel("Leistung $P\ [\mathrm{mA}] $", family="serif", fontsize="14")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("Grafiken/Leistung_100mA.pdf")
plt.show()
plt.clf()


## Print Funktionen