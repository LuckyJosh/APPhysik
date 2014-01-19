# -*- coding: utf-8 -*-
"""
Created on Fri Nov 08 23:56:02 2013

@author: Josh

"""

from __future__ import (print_function,
                        division,
                        unicode_literals,
                        absolute_import)
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit
from scipy import signal
from sympy import *
import uncertainties as unc
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)

sys.path.append("..\_globales\python")
import latextables as lxtabs


def H(x, a, b):
    return a*np.exp(-b * x)


def A(x, RC):
    return 1/np.sqrt(1 + 4 * const.pi**2 * x**2 * RC**2)


def P(x, RC):
    return np.arctan(RC * 2 * const.pi * x)


def Ap(p, RC):
    return np.cos(p)/(RC)

Ulog = unc.wrap(np.log)


# Laden der Messdaten der Aufladung
t_auf, U_auf = np.loadtxt("Messdaten/Aufladen.txt", unpack=True)


# Laden der Messdaten der Entladung
t_ent, U_ent = np.loadtxt("Messdaten/Entladen.txt", unpack=True)


# Fehlerbehaftete Messwerte (Fehler für Zeit vernachlässigbar)
U_err = 1  # [V]

uU_ent = unp.uarray(U_ent, [U_err]*len(U_ent))

# Regression der Messwerte
popt1, pcov1 = curve_fit(H, t_ent, noms(uU_ent))

# Fehler der Parameter
errors1 = np.sqrt(np.diag(pcov1))
um1 = ufloat(popt1[1], errors1[1])
ub1 = ufloat(popt1[0], errors1[0])

# Berechnung der Zeitkonstante
RC1 =  1/um1

# Plot der Messwerte und Reressionskurve
t = np.linspace(-0.0001, 0.0009, num=1000)
plt.clf()
plt.grid()
plt.xlim(-0.0001, 0.0009)
plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter
                                    (lambda x, _: float(x * 1e3)))
plt.ylim(1e00,  1e03)
plt.xlabel("Zeit $t\,[\mathrm{ms}]$")
plt.ylabel("Kondensatorspannung $U_{C}\,[\mathrm{V}]$")
plt.yscale("log")
plt.errorbar(t_ent, noms(uU_ent), yerr=stds(uU_ent),
             fmt="rx", label="Messdaten")
plt.plot(t, H(t, *popt1), color="gray", label="Regressionsgerade")

plt.legend(loc="best")
plt.tight_layout()
plt.savefig("Grafiken/Entladung.pdf")
# Generatorspannung U0
U0 = 80  # [V]
# TODO: Generatorinnenwiederstand berücksichtigen

# Laden der Frequenzabhängigen Amplituden
f1, Amps1 = np.loadtxt("Messdaten/Frequenzen_Amplituden_1.txt", unpack=True)
f2, Amps2 = np.loadtxt("Messdaten/Frequenzen_Amplituden_2.txt", unpack=True)

# Laden der Fehler
Amps1_err, Amps2_err = np.loadtxt("Messdaten/Amplituden_Fehler.txt",
                                  unpack=True)
# Fehlerbehaftete Messwerte
uAmps1 = unp.uarray(Amps1, Amps1_err)
uAmps2 = unp.uarray(Amps2, Amps2_err)
uAmps = np.concatenate((uAmps1, uAmps2))

# Skalierte Amplituden
uAmps_U0 = uAmps/U0

# Verbindung der Frequenzen
f = np.concatenate((f1, f2))

# Regeression der Messwerte
popt2, pcov2 = curve_fit(A, f, noms(uAmps_U0), sigma=stds(uAmps_U0))
F = np.linspace(0, 1e05, num=100000)

# Fehler
errors2 = np.sqrt(np.diag(pcov2))
uc2 = ufloat(popt2[0], errors2[0])

# Plot der Messwerte und Reressionskurve

plt.clf()
plt.grid()
plt.xscale("log")
plt.xlim(1e01, 2e05)
#plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter
#                                    (lambda x, _: float(x * 1e3)))
#plt.ylim(1e00,  1e03)
plt.xlabel("Frequenz $f\,[\mathrm{Hz}]$")
plt.ylabel(r"relativ Amplitude $\frac{A(f)}{U_{0}}$")

plt.errorbar(f, noms(uAmps_U0), yerr=stds(uAmps_U0),
             fmt="rx", label="Messdaten")
plt.plot(F, A(F, *popt2), color="gray", label="Regressionskurve")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("Grafiken/Amplitude.pdf")
# Laden der Frequenzen und der Gangunterschiede
f3, a = np.loadtxt("Messdaten/Gangunterschied.txt", unpack=True)

# Berechnung der Periodendauer b
b = 1/f3

# Berechnung der Phasendifferenz
p = 2 * const.pi * a/b
p_grad = 360 * a/b
# Regeression der Messwerte
popt3, pcov3 = curve_fit(P, f3, p)
F = np.linspace(0, 1e06, num=1000000)

# Fehler
errors3 = np.sqrt(np.diag(pcov3))
ud3 = ufloat(popt3[0], errors3[0])

# Plot der Messwerte und Reressionskurve
t = np.linspace(-0.0001, 0.0009, num=1000)
plt.clf()
plt.grid()
plt.xscale("log")
plt.xlim(1e01, 1e06)
plt.tick_params("y", labelsize=18)
plt.yticks((pi/16, pi/8, 3 * pi/16, pi/4, 5 * pi/16,
            3 * pi/8, 7 * pi/16, pi/2),
           (r"$\frac{\pi}{16}$", r"$\frac{\pi}{8}$", r"$\frac{3\pi}{16}$",
            r"$\frac{\pi}{4}$", r"$\frac{5\pi}{16}$", r"$\frac{3\pi}{8}$",
            r"$\frac{7\pi}{16}$", r"$\frac{\pi}{2}$"))
#plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter
#                                    (lambda x, _: float(x * 1e3)))
#plt.ylim(1e00,  1e03)
plt.xlabel("Frequenz $f\,[\mathrm{Hz}]$")
plt.ylabel(r"Phasendifferenz ${\varphi}$")

plt.plot(f3, p, "rx", label="Messdaten")
plt.plot(F, P(F, *popt3), color="gray", label="Regressionskurve")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("Grafiken/Phasendifferenz.pdf")
# Polarer Plot


Amps_U0 = np.array([noms(uAmps_U0[4]), noms(uAmps_U0[5]),
                    noms(uAmps_U0[6]), noms(uAmps_U0[9]),
                    noms(uAmps_U0[10]), noms(uAmps_U0[11]),
                    noms(uAmps_U0[14]), noms(uAmps_U0[15]),
                    noms(uAmps_U0[16]), noms(uAmps_U0[18])])

Phi = np.array([p[0], p[2], p[4], p[5], p[7], p[9], p[10], p[12], p[14], p[15]])

Ph = np.linspace(0, const.pi/2)
freqs = np.linspace(1, 1e02, num=1e02)

plt.clf()





#plt.yticks((pi/16, pi/8, 3 * pi/16, pi/4, 5 * pi/16,
#            3 * pi/8, 7 * pi/16, pi/2),
#           (r"$\frac{\pi}{16}$", r"$\frac{\pi}{8}$", r"$\frac{3\pi}{16}$",
#            r"$\frac{\pi}{4}$", r"$\frac{5\pi}{16}$", r"$\frac{3\pi}{8}$",
#            r"$\frac{7\pi}{16}$", r"$\frac{\pi}{2}$"))
#plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter
#                                    (lambda x, _: float(x * 1e3)))

#plt.xlabel("Frequenz $f\,[\mathrm{Hz}]$")
#plt.ylabel(r"Phasendifferenz ${\varphi}$")

plt.polar(Phi, Amps_U0, "rx", label="Messdaten")
plt.polar(Ph, np.cos(Ph), color="gray", label="Theoriekurve")

plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("Grafiken/Amplitude_Polar.pdf")


#f =  open("Daten/Tabelle_Entladen.tex", "w")
#
#f.write(lxtabs.toTable([t_ent * 1e03, uU_ent],
#        col_titles=["Zeit", "Kondensatorspannung"],
#        col_syms=[r"t", r"{U_{C}}"],
#        col_units=[r"\milli\second", r"\volt"],
#        fmt=["c", "c"],
#        cap="Kondensatorspannung zur Zeit $t$ nach Beginn der Entladung",
#        label="Auswertung_Entladen"))
#
#f.close()

#File = open("Daten/Tabelle_Amplituden.tex", "w")
#
#File.write(lxtabs.toTable([f, uAmps, uAmps_U0],
#        col_titles=["Frequenz", "Amplitude", "normierte Amplitude"],
#        col_syms=[r"f", r"A(f)", r"\tfrac{A(f)}{U_{0}}"],
#        col_units=[r"\hertz", r"\volt", r""],
#        fmt=["c", "c", "c"],
#        cap="Amplitude und normierte Amplitude der Kondensatorspannung in" +
#            " Abhängikeit der Frequenz",
#        label="Auswertung_Amplitude"))
#
#File.close()

#File = open("Daten/Tabelle_Phasendifferenz.tex", "w")
#
#File.write(lxtabs.toTable([f3, p_grad],
#        col_titles=["Frequenz", "Phasendifferenz"],
#        col_syms=[r"f", r"\varphi(f)"],
#        col_units=[r"\hertz", r"\degree"],
#        fmt=["c", "c"],
#        cap="Phasendifferenz der Kondensator- und Generatorspannung in" +
#            " Abhängikeit der Frequenz",
#        label="Auswertung_Phasendifferenz"))
#
#File.close()




