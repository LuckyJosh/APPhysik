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




def temp2press(T):
    return 5.5e07 * np.exp(-6876/(T + 273.15))


def press2dist(P):
    return 0.0029 / P

# Laden der Temperaturen
T = np.loadtxt("Messdaten/Temperaturen.txt")
for t in T:
    P = temp2press(t)
    print("Dampfdruck T=", str(t),":", P)
    print("mittlere freie Weglänge T=", str(t), ":", press2dist(P))



#=============================================================================
class Raumtemperatur:
     pass
#=============================================================================

# Laden der Messdaten
U_1, I_1 = np.loadtxt("Messdaten/I_aU_a_I.txt", unpack=True)

#dI_1 = [(I_1[i+1] - I_1[i])/(U_1[i+1] - U_1[i]) for i in range(len(I_1)-1)]
dI_1 = [(I_1[i] - I_1[i+1]) for i in range(len(I_1)-1)]

print("Maximum:", U_1[np.where(dI_1 == max(dI_1))[0][0]], "/", max(dI_1))
print("Kontaktpotenzial:", 10 - U_1[np.where(dI_1 == max(dI_1))[0][0]])


plt.plot(U_1[:-1], dI_1, label="differenzielle Energieverteilung")
plt.grid()
#plt.show()


#=============================================================================
class Grad_150:
     pass
#=============================================================================

# Laden der Messdaten
U_2, I_2 = np.loadtxt("Messdaten/I_aU_a_II.txt", unpack=True)

#dI_2 = [(I_2[i+1] - I_2[i])/(U_2[i+1] - U_2[i]) for i in range(len(I_2)-1)]
dI_2 = [(I_2[i] - I_2[i+1]) for i in range(len(I_2)-1)]


plt.plot(U_2[:-1], dI_2, label="differenzielle Energieverteilung")
plt.plot(U_2[:-1], dI_2, "rx", label="differenzielle Energieverteilung")
plt.grid()
#plt.show()


#=============================================================================
class FranckHertz:
     pass
#=============================================================================
# Laden der Messwerte
U_3 = np.loadtxt("Messdaten/FranckHertzKurve.txt")  # cm

U_3 *= 35/16  # V
dU_3 = [U_3[i+1] - U_3[i] for i in range(len(U_3) - 1)]
Q_dU_3 = Quantity(dU_3[1:])
print(U_3, dU_3, Q_dU_3.avr ,dU_3[0]-Q_dU_3.avr)






## Print Funktionen