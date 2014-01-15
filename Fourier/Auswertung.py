# -*- coding: utf-8 -*-
"""
Created on Fri Nov 08 23:56:02 2013

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

sys.path.append("..\globales\python")
import latextables as lxtabs


PRINT = True
TABS = True

def NullFilter(arr):
    i = 0
    M = np.alen(np.where(arr == 0)[0])
    N = np.alen(arr) - M
    arr_new = np.zeros(N)
    for n in range(np.alen(arr)):
        if arr[n] != 0:
            arr_new[i] = arr[n]
            i += 1
    return arr_new

# Fourier Koeffitienten
def bn_rect(A, n):
    return 4 * A / (n * const.pi)


def bn_tri(A, n):
    return 4 * A / (n * const.pi)**2


def bn_saw(A, n):
    return 2 * A / (n * const.pi)

# Fourier Analyse

# Amplitude
U = 2  # [V]

## Rechteck
amps_rect = np.zeros(12)
for n in range(12):
    n += 1
    if n % 2 != 0:
        amps_rect[n] = bn_rect(U, n)

# Filtern der Nullen
amps_rect = NullFilter(amps_rect)

## Dreieck
amps_tri = np.zeros(11)
for n in range(10):
    n += 1
    if n % 2 != 0:
        amps_tri[n] = bn_tri(U, n)

# Filtern der Nullen
amps_tri = NullFilter(amps_tri)

## Sägezahn
amps_saw = np.zeros(8)
for n in range(7):
    n += 1
    amps_saw[n] = bn_saw(U, n)

# Filtern der Nullen
amps_saw = NullFilter(amps_saw)

# Fourier- Synthese

B_1 = 0.8  # [V]
A_rect = 0.2 * const.pi
A_tri = 0.2 * const.pi**2
A_saw = 0.4 * const.pi

## Rechteck
amps_rect_syn = np.zeros(10)
for n in range(10):
    n += 1
    if n % 2 != 0:
        amps_rect_syn[n] = bn_rect(A_rect, n)

# Filtern der Nullen
#amps_rect_syn = NullFilter(amps_rect_syn)

## Dreieck
amps_tri_syn = np.zeros(10)
for n in range(10):
    n += 1
    if n % 2 != 0:
        amps_tri_syn[n] = bn_tri(A_tri, n)

# Filtern der Nullen
#amps_tri_syn = NullFilter(amps_tri_syn)

## Sägezahn
amps_saw_syn = np.zeros(11)
for n in range(10):
    n += 1
    amps_saw_syn[n] = bn_saw(A_saw, n)

# Filtern der Nullen
#amps_saw_syn = NullFilter(amps_saw_syn)




# Laden der Messdaten

## Rechteck
Amps_rect = np.loadtxt("Messdaten/FFT_Rechteck.txt", unpack=True)

## Dreieck
Amps_tri = np.loadtxt("Messdaten/FFT_Dreieck.txt", unpack=True)

## Sägezahn
Amps_saw = np.loadtxt("Messdaten/FFT_Sägezahn.txt", unpack=True)


# Relative Abweichung der Messung von der Theorie

## Rechteck
Amps_rect_rel = np.abs(1 - Amps_rect/amps_rect)
## Dreieck
Amps_tri_rel = np.abs(1 - Amps_tri/amps_tri)
## Sägezahn
Amps_saw_rel = np.abs(1 - Amps_saw/amps_saw)

# Ungerade Frequenzen
F = 100  # [Hz]
freqs = np.array([1, 3, 5, 7, 9, 11])
freqs *= F

# Die N's
N = np.arange(1, 11, 1)


## Print Funktionen
if PRINT:
    print("Berechnete Werte:\n")
    print("Analyse Koeffitienten für...\n\t" +
          "...Rechteck:\n\t{}\n\t".format(amps_rect) +
          "...Dreieck:\n\t{}\n\t".format(amps_tri) +
          "...Sägezahn:\n\t{}".format(amps_saw))

    print("\nSyntese Koeffitienten für...\n\t" +
          "...Rechteck:\n\t{}\n\t".format(amps_rect_syn) +
          "...Dreieck:\n\t{}\n\t".format(amps_tri_syn) +
          "...Sägezahn:\n\t{}".format(amps_saw_syn))
    print("\nGemessene Werte:\n")
    print("Analyse Koeffitienten für...\n\t" +
          "...Rechteck:\n\t{}\n\t".format(Amps_rect) +
          "...Dreieck:\n\t{}\n\t".format(Amps_tri) +
          "...Sägezahn:\n\t{}".format(Amps_saw))
    print("\nrelative Abweichungen:...\n")
    print("\t...Rechteck:\n\t{}\n\t".format(Amps_rect_rel) +
          "...Dreieck:\n\t{}\n\t".format(Amps_tri_rel) +
          "...Sägezahn:\n\t{}".format(Amps_saw_rel))

#if TABS:
#    f = open("Daten/Tabelle_Analyse1.tex", "w")
#    f.write(lxtabs.toTable([freqs, Amps_rect, amps_rect,
#                            Amps_tri, amps_tri],
#        col_titles=["Frequenzen",
#                    "Gemessene Amplitude",
#                    "Berechnete Amplitude",
#                    "Gemessene Amplitude",
#                    "Berechnete Amplitude"],
#        col_syms=[r"\nu", r"b_{n}", r"b_{n}", r"b_{n}", r"b_{n}"],
#        col_units=[r"\hertz", r"\volt", r"\volt", r"\volt", r"\volt"],
#        fmt=["c", "c", "c", "c", "c"],
#        cap="Gemessene und Berechnete Amplituden der Oberschwingung für " +
#        "Recht- und Dreieckspannung",
#        label="Analyse1"))
#
#    f.close()
#
#    f = open("Daten/Tabelle_Synthese.tex", "w")
#    f.write(lxtabs.toTable([np.append(amps_rect_syn, 0),
#                            np.append(amps_tri_syn, 0),
#                            amps_saw_syn],
#        col_titles=["Rechteck Amplitude",
#                    "Dreieck Amplitude",
#                    "Sägezahn Amplitude"],
#        col_syms=[r"b_{n,r}", r"b_{n,d}", r"b_{n,s}"],
#        col_units=[r"\volt", r"\volt", r"\volt"],
#        fmt=["c", "c", "c"],
#        cap="Zur Synthese verwandte Amplituden der ersten 10 Oberwellen",
#        label="Synthese"))
#
#    f.close()

