# -*- coding: utf-8 -*-
"""
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
from sympy import *
import uncertainties as unc
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)

from aputils.utils import Quantity, ErrorEquation
from aputils.latextables.tables import Table

Uexp = unc.wrap(np.exp)


def temp2press(T):
    return 5.5e07 * Uexp(-6876/(T + 273.15))


def press2dist(P):
    return 0.0029 / P

# Laden der Temperaturen
T = np.loadtxt("Messdaten/Temperaturen.txt")
t_err = 0.1

T_err = unp.uarray(T, [t_err]*len(T))

for t in T_err:
    P = temp2press(t)
    print("Dampfdruck T=", str(t),"=" , str(t + 273.15) ,":", P)
    print("mittlere freie Weglänge T=", str(t), str(t + 273.15) ,":", press2dist(P))



#=============================================================================
class Raumtemperatur:
     pass
#=============================================================================

# Laden der Messdaten
U_1, I_1 = np.loadtxt("Messdaten/I_aU_a_I.txt", unpack=True)

# Fehler
u_err = 0.05  # Volt pro Millimeter
i_err = 0.1   # Zentimeter

U_1_err = unp.uarray(U_1, [u_err]*len(U_1))
I_1_err = unp.uarray(I_1, [i_err]*len(U_1))


dU_1_err = [(U_1_err[i+1] - U_1_err[i]) for i in range(len(I_1_err)-1)]


dI_1_err = [(I_1_err[i] - I_1_err[i+1]) for i in range(len(I_1)-1)]

print("Maximum:", U_1_err[np.where(noms(dI_1_err) == max(noms(dI_1_err)))[0][0]], "/", max(dI_1_err))
print("Kontaktpotenzial:", ufloat(11, 1) - U_1_err[np.where(noms(dI_1_err) == max(noms(dI_1_err)))[0][0]])
K_1_err = ufloat(11, 1) - U_1_err[np.where(noms(dI_1_err) == max(noms(dI_1_err)))[0][0]]


plt.plot(noms(U_1_err[:-1]), noms(dI_1_err), label="differenzielle\nEnergieverteilung")
plt.stem([noms(U_1_err[np.where(noms(dI_1_err) == max(noms(dI_1_err)))[0][0]])], [max(noms(dI_1_err))],
         linefmt="k--", markerfmt="ko",
         label="max. Änderung ({}|{})".format(U_1[np.where(noms(dI_1_err) == max(noms(dI_1_err)))[0][0]],
                                              max(noms(dI_1_err))))
plt.grid()
plt.xlabel(r"Bremsspannung $U_{A}$")
plt.ylabel("Änderung des Auffängerstroms\n $I_{A}(U_{A}) - I_{A}(U_{A} + \Delta U_{A})$")
plt.legend(loc="upper left")
#plt.show()
plt.savefig("Grafiken/Diff_EVerteilung_20.pdf")
plt.clf()

I_1_err = np.append(I_1_err, 1337)
dI_1_err = np.append(dI_1_err, 1337)
dI_1_err = np.append(dI_1_err, 1337)
U_1_err = np.append(U_1_err, 1337)
dU_1_err = np.append(dU_1_err, 1337)
dU_1_err = np.append(dU_1_err, 1337)

# Tabelle
Tab_Diff_1 = Table(siunitx=True,)
Tab_Diff_1.label("Auswertung_Diff_Energie_Verteilung_20C")
Tab_Diff_1.caption("Messwerte zur Bestimmung der differntiellen Energieverteilung bei Raumtemperatur")
Tab_Diff_1.layout(seperator="column", title_row_seperator="double", border=True)
Tab_Diff_1.addColumn(U_1_err[:22], title="Bremsspannung", symbol="U_{A}", unit=r"\volt")
Tab_Diff_1.addColumn(dU_1_err[:22], title="$\Delta$ Bremsspannung", symbol="\Delta U_{A}", unit=r"\volt")
Tab_Diff_1.addColumn(I_1_err[:22], title="Auffängerstrom", symbol="\propto I_{A}(U_{A})")
Tab_Diff_1.addColumn(dI_1_err[:22], title="$\Delta$ Auffängerstrom", symbol="\propto \Delta I_{A}(U_{A})")
Tab_Diff_1.addColumn(U_1_err[22:], title="Bremsspannung", symbol="U_{A}", unit=r"\volt")
Tab_Diff_1.addColumn(dU_1_err[22:], title="$\Delta$ Bremsspannung", symbol="\Delta U_{A}", unit=r"\volt")
Tab_Diff_1.addColumn(I_1_err[22:], title="Auffängerstrom", symbol="\propto I_{A}(U_{A})")
Tab_Diff_1.addColumn(dI_1_err[22:], title="$\Delta$ Auffängerstrom", symbol="\propto \Delta I_{A}(U_{A})")
#Tab_Diff_1.show(quiet=False)
#Tab_Diff_1.save("Tabellen/Diff_EVerteilung_20C.tex")
#=============================================================================
class Grad_150:
     pass
#=============================================================================

# Laden der Messdaten
U_2, I_2 = np.loadtxt("Messdaten/I_aU_a_II.txt", unpack=True)

# Fehler
U_2_err = unp.uarray(U_2, [u_err]*len(U_2))
I_2_err = unp.uarray(I_2, [i_err]*len(I_2))


dU_2_err = [(U_2_err[i+1] - U_2_err[i]) for i in range(len(U_2_err)-1)]
dI_2_err = [(I_2_err[i] - I_2_err[i+1]) for i in range(len(I_2_err)-1)]


plt.plot(noms(U_2_err[:-1]), noms(dI_2_err), label="differenzielle\nEnergieverteilung")
#plt.plot(U_2[:-1], dI_2, "rx", label="differenzielle Energieverteilung")
plt.grid()
plt.xlabel(r"Bremsspannung $U_{A}$")
plt.ylabel("Änderung des Auffängerstroms\n $I_{A}(U_{A}) - I_{A}(U_{A} + \Delta U_{A})$")
plt.legend(loc="upper right")
#plt.show()
plt.savefig("Grafiken/Diff_EVerteilung_150.pdf")

#plt.show()
plt.clf()


dI_2_err = np.append(dI_2_err, 1337)
dU_2_err = np.append(dU_2_err, 1337)

# Tabelle
Tab_Diff_2 = Table(siunitx=True,)
Tab_Diff_2.label("Auswertung_Diff_Energie_Verteilung_150C")
Tab_Diff_2.caption("Messwerte zur Bestimmung der differntiellen Energieverteilung bei \\SI{150}{\\degreeCelsius}")
Tab_Diff_2.layout(seperator="column", title_row_seperator="double", border=True)
Tab_Diff_2.addColumn(U_2_err[:18], title="Bremsspannung", symbol="U_{A}", unit=r"\volt")
Tab_Diff_2.addColumn(dU_2_err[:18], title="$\Delta$ Bremsspannung", symbol="\Delta U_{A}", unit=r"\volt")
Tab_Diff_2.addColumn(I_2_err[:18], title="Auffängerstrom", symbol="\propto I_{A}(U_{A})")
Tab_Diff_2.addColumn(dI_2_err[:18], title="$\Delta$ Auffängerstrom", symbol="\propto \Delta I_{A}(U_{A})")
Tab_Diff_2.addColumn(U_2_err[18:], title="Bremsspannung", symbol="U_{A}", unit=r"\volt")
Tab_Diff_2.addColumn(dU_2_err[18:], title="$\Delta$ Bremsspannung", symbol="\Delta U_{A}", unit=r"\volt")
Tab_Diff_2.addColumn(I_2_err[18:], title="Auffängerstrom", symbol="\propto I_{A}(U_{A})")
Tab_Diff_2.addColumn(dI_2_err[18:], title="$\Delta$ Auffängerstrom", symbol="\propto \Delta I_{A}(U_{A})")
#Tab_Diff_2.show(quiet=True)
#Tab_Diff_2.save("Tabellen/Diff_EVerteilung_150C.tex")
#=============================================================================
class FranckHertz:
     pass
#=============================================================================
# Laden der Messwerte
U_3 = np.loadtxt("Messdaten/FranckHertzKurve.txt")  # cm

#Fehler
u_err = 0.2
U_3_err = unp.uarray(U_3, [u_err]*len(U_3))


U_3_err *= 35/16  # V
dU_3_err = [U_3_err[i+1] - U_3_err[i] for i in range(len(U_3_err) - 1)]
Q_dU_3 = Quantity(noms(dU_3_err)[1:])
print(Q_dU_3.avr_err, (const.h*const.speed_of_light/const.elementary_charge)/Q_dU_3.avr_err ,dU_3_err[0]-Q_dU_3.avr_err)
K_2_err = dU_3_err[0]-Q_dU_3.avr_err

dU_3_err = np.append(dU_3_err, 1337)
# Tabelle
Tab_Diff_3 = Table(siunitx=True,)
Tab_Diff_3.label("Auswertung_Diff_Energie_Verteilung_150C")
Tab_Diff_3.caption("Messwerte zur Bestimmung der differntiellen Energieverteilung bei \\SI{150}{\\degreeCelsius}")
Tab_Diff_3.layout(seperator="column", title_row_seperator="double", border=True)
Tab_Diff_3.addColumn(U_3_err[:], title="maximal Stellen", symbol="U_{A,max,i}", unit=r"\volt")
Tab_Diff_3.addColumn(dU_3_err[:], title="$\Delta$ maximal Stellen", symbol="U_{A,max,i+1} - U_{A,max,i+1}", unit=r"\volt")
Tab_Diff_3.addColumn(dU_3_err[:], title="Anregungsenergie", symbol="E", unit="\eV")
#Tab_Diff_3.show(quiet=True)
#Tab_Diff_3.save("Tabellen/FranckHertz_Kurve.tex")

#=============================================================================
class Ionisierungsspannung:
     pass
#=============================================================================
K_avr = np.mean([K_1_err, K_2_err])
print(K_avr)

U_0 = ufloat(12.5, 0.2)
U_ion = U_0 - K_avr
print(U_ion)






