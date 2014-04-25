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
#mpl.use("Qt4Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit
import sympy as sp
import sys
import uncertainties as unc
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)


from aputils.utils import Quantity, ErrorEquation, OutputFile
from aputils.latextables.tables import Table

# Erzeugen eines logs des stdout
sys.stdout = OutputFile("Daten/log.txt")

PRINT = True

#==============================================================================
#
# Elektronen im Elektrischen Feld
#
#==============================================================================

# Import der Ablenkspannungen
U_D = np.array(np.loadtxt("Messdaten/Ablenkspannung.txt", unpack=True))

# Import des Fehlers
u_d_err = np.loadtxt("Messdaten/Fehler_Ablenkspannung.txt")

# Erzeugung der einzelen Messreihen, löschen der unbesetzten Stellen
U_D_1 = U_D[0]
U_D_2 = U_D[1]
U_D_3 = U_D[2]
U_D_4 = np.delete(U_D[3], (np.where(U_D[3] == 0))[0])
U_D_5 = np.delete(U_D[4], (np.where(U_D[4] == 0))[0])


# Erstellen der fehlerbehafteten Größen
U_D_1_err = unp.uarray(U_D_1, len(U_D_1)*[u_d_err])
U_D_2_err = unp.uarray(U_D_2, len(U_D_2)*[u_d_err])
U_D_3_err = unp.uarray(U_D_3, len(U_D_3)*[u_d_err])
U_D_4_err = unp.uarray(U_D_4, len(U_D_4)*[u_d_err])
U_D_5_err = unp.uarray(U_D_5, len(U_D_5)*[u_d_err])


# Import der Beschleunigungsspannungen
U_B = np.loadtxt("Messdaten/Beschleunigungsspannung_E.txt", unpack=True)

# Import des Fehlers
u_b_err = np.loadtxt("Messdaten/Fehler_Beschleunigungsspannung.txt")

# Erzeugen der fehlerbehafteten Größen
U_B_err = unp.uarray(U_B, len(U_B)*[u_b_err])

# Berechnung der Distanzen
D = np.arange(0, 9, 1) * (const.inch/4) * 1e02  # cm

#Fehler der Distanzen
d_err = 0.2 * (const.inch/4) * 1e02  # cm

# Berechnung der fehlerbehafteten Größen
D_err = unp.uarray(D, [d_err]*len(D))

##Erstellen der Tabelle der Messwerte
#T_E_1 = Table(siunitx=True)
#T_E_1.caption("Messdaten zur Bestimmung des Zusammenhangs zwischen $U_d$ und $D$")
#T_E_1.label("Auswertung_Messdaten_I")
#T_E_1.layout(seperator="column", border=True)
#T_E_1.addColumn(U_B_err)
##T_E_1.addColumn(U_D_1_err, title="1", symbol="")
##T_E_1.addColumn(U_D_2_err, title="2", symbol="")
##T_E_1.addColumn(U_D_3_err, title="3", symbol="")
##T_E_1.addColumn(np.concatenate((U_D_4_err, [0])), title="4", symbol="")
##T_E_1.addColumn(np.concatenate((U_D_5_err, [0,0])), title="5", symbol="")
##T_E_1.addColumn(D_err, title="Verschiebung", symbol="D", unit=r"\centi\meter")
##T_E_1.save("Tabellen/Messwerte_I.tex")
##T_E_1.show()
#T_E_1.save("Tabellen/Messwerte_I_Ub.tex")


# Erstellen des Plots zur Bestimmung der Empfindlichkeit

# Definition einer Geraden
def gerade(x, m, b):
    return m * x + b


#==============================================================================
class EMessreiheI:
    pass
#==============================================================================

## Berechnung der Ausgleichsgerade
popt_1, pcov_1 = curve_fit(gerade, noms(U_D_1_err),
                           noms(D_err), sigma=stds(D_err))
error_1 = np.sqrt(np.diag(pcov_1))
param_m_1 = ufloat(popt_1[0], error_1[0])
param_b_1 = ufloat(popt_1[1], error_1[1])

print("Ausgleichsgerade I: m, b:", param_m_1, param_b_1) if PRINT else print()


## Messdaten
plt.errorbar(noms(U_D_1_err), noms(D_err),
             xerr=stds(U_D_1_err), yerr=stds(D_err),
             fmt="rx", label="Messwerte")

## Ausgleichsgrade
X = np.linspace(-40, 40, 200)
plt.plot(X, gerade(X, *popt_1), color="gray", label="Regressionsgeraden")

## Ploteinstellungen
plt.grid()
plt.xlim(-40, 25)
plt.ylim(-1, 6)
plt.xlabel(r"Ablenkspannung $U_d \ [\mathrm{V}]$", fontsize=14, family='serif')
plt.ylabel(r"Verschiebung $D \ [\mathrm{cm}]$", fontsize=14, family='serif')
plt.legend(loc="best")
plt.tight_layout()
#plt.show()
plt.savefig("Grafiken/EFeld_Messreihe_I.pdf")
plt.clf()

#==============================================================================
class EMessreiheII:
    pass
#==============================================================================

## Berechnung der Ausgleichsgerade
popt_2, pcov_2 = curve_fit(gerade, noms(U_D_2_err),
                           noms(D_err), sigma=stds(D_err))
error_2 = np.sqrt(np.diag(pcov_2))
param_m_2 = ufloat(popt_2[0], error_2[0])
param_b_2 = ufloat(popt_2[1], error_2[1])

print("Ausgleichsgerade II: m, b:", param_m_2, param_b_2) if PRINT else print()


## Messdaten
plt.errorbar(noms(U_D_2_err), noms(D_err),
             xerr=stds(U_D_2_err), yerr=stds(D_err),
             fmt="rx", label="Messwerte")

## Ausgleichsgrade
X = np.linspace(-40, 40, 200)
plt.plot(X, gerade(X, *popt_2), color="gray", label="Regressionsgerade")

## Ploteinstellungen
plt.grid()
plt.xlim(-40, 25)
plt.ylim(-1, 6)
plt.xlabel(r"Ablenkspannung $U_d \ [\mathrm{V}]$", fontsize=14, family='serif')
plt.ylabel(r"Verschiebung $D \ [\mathrm{cm}]$", fontsize=14, family='serif')
plt.legend(loc="best")
plt.tight_layout()
#plt.show()
plt.savefig("Grafiken/EFeld_Messreihe_II.pdf")
plt.clf()

#==============================================================================
class EMessreiheIII:
    pass
#==============================================================================

## Berechnung der Ausgleichsgerade
popt_3, pcov_3 = curve_fit(gerade, noms(U_D_3_err),
                           noms(D_err), sigma=stds(D_err))
error_3 = np.sqrt(np.diag(pcov_3))
param_m_3 = ufloat(popt_3[0], error_3[0])
param_b_3 = ufloat(popt_3[1], error_3[1])

print("Ausgleichsgerade III: m, b:", param_m_3, param_b_3) if PRINT else print()


## Messdaten
plt.errorbar(noms(U_D_3_err), noms(D_err),
             xerr=stds(U_D_3_err), yerr=stds(D_err),
             fmt="rx", label="Messreihe")

## Ausgleichsgrade
X = np.linspace(-40, 40, 200)
plt.plot(X, gerade(X, *popt_3), color="gray", label="Regressionsgerade")

## Ploteinstellungen
plt.grid()
plt.xlim(-40, 25)
plt.ylim(-1, 6)
plt.xlabel(r"Ablenkspannung $U_d \ [\mathrm{V}]$", fontsize=14, family='serif')
plt.ylabel(r"Verschiebung $D \ [\mathrm{cm}]$", fontsize=14, family='serif')
plt.legend(loc="best")
plt.tight_layout()
#plt.show()
plt.savefig("Grafiken/EFeld_Messreihe_III.pdf")
plt.clf()

#==============================================================================
class EMessreiheIV:
    pass
#==============================================================================

## Berechnung der Ausgleichsgerade
popt_4, pcov_4 = curve_fit(gerade, noms(U_D_4_err),
                           noms(D_err[1:]), sigma=stds(D_err[1:]))
error_4 = np.sqrt(np.diag(pcov_4))
param_m_4 = ufloat(popt_4[0], error_4[0])
param_b_4 = ufloat(popt_4[1], error_4[1])

print("Ausgleichsgerade IV: m, b:", param_m_4, param_b_4) if PRINT else print()


## Messdaten
plt.errorbar(noms(U_D_4_err), noms(D_err[1:]),
             xerr=stds(U_D_4_err), yerr=stds(D_err[1:]),
             fmt="rx", label="Messwerte")

## Ausgleichsgrade
X = np.linspace(-40, 40, 200)
plt.plot(X, gerade(X, *popt_4), color="gray", label="Regressionsgerade")


## Ploteinstellungen
plt.grid()
plt.xlim(-40, 25)
plt.ylim(-1, 6)
plt.xlabel(r"Ablenkspannung $U_d \ [\mathrm{V}]$", fontsize=14, family='serif')
plt.ylabel(r"Verschiebung $D \ [\mathrm{cm}]$", fontsize=14, family='serif')
plt.legend(loc="best")
plt.tight_layout()
#plt.show()
plt.savefig("Grafiken/EFeld_Messreihe_IV.pdf")
plt.clf()

#==============================================================================
class EMessreiheV:
    pass
#==============================================================================

## Berechnung der Ausgleichsgerade
popt_5, pcov_5 = curve_fit(gerade, noms(U_D_5_err),
                           noms(D_err[2:]), sigma=stds(D_err[2:]))
error_5 = np.sqrt(np.diag(pcov_5))
param_m_5 = ufloat(popt_5[0], error_5[0])
param_b_5 = ufloat(popt_5[1], error_5[1])

print("Ausgleichsgerade V: m, b:", param_m_5, param_b_5) if PRINT else print()


## Messdaten
plt.errorbar(noms(U_D_5_err), noms(D_err[2:]),
             xerr=stds(U_D_5_err), yerr=stds(D_err[2:]),
             fmt="rx", label="Messwerte")

## Ausgleichsgrade
X = np.linspace(-40, 40, 200)
plt.plot(X, gerade(X, *popt_5), color="gray", label="Regressionsgerade")

## Ploteinstellungen
plt.grid()
plt.xlim(-40, 25)
plt.ylim(-1, 6)
plt.xlabel(r"Ablenkspannung $U_d \ [\mathrm{V}]$", fontsize=14, family='serif')
plt.ylabel(r"Verschiebung $D \ [\mathrm{cm}]$", fontsize=14, family='serif')
plt.legend(loc="best")
plt.tight_layout()
#plt.show()
plt.savefig("Grafiken/EFeld_Messreihe_V.pdf")
plt.clf()

# Erstellen des Plots Empfindlichkeiten gegen 1/Beschleunigungsspannung

## Speichern der Empfindlichleiten
Empf_err = np.array([param_m_1, param_m_2,
                     param_m_3, param_m_4,
                     param_m_5])
## Speichen der Abschnitte
param_b_err = np.array([param_b_1, param_b_2,
                        param_b_3, param_b_4,
                        param_b_5])

# Speichern der Fitparamter in Tabelle
T_params_E = Table(siunitx=True)
T_params_E.label("Auswertung_Parameter_E")
T_params_E.caption("Fit-Parameter der Daten aus den fünf Messreihen")
T_params_E.layout(seperator="column", title_row_seperator="double",
                  border=True)
T_params_E.addColumn(Empf_err, title="Steigung",
                     symbol="m", unit=r"\meter\per\volt")
T_params_E.addColumn(param_b_err, title="y-Achsenabschnitt",
                     symbol="b", unit=r"\meter")
#T_params_E.show()
#T_params_E.save("Tabellen/Parameter_E.tex")


#==============================================================================
class EMessreiheVI:
    pass
#==============================================================================
## Berechnung der Regressionsgraden
popt_6, pcov_6 = curve_fit(gerade, noms(1/U_B_err), noms(Empf_err),
                           sigma=stds(Empf_err))
error_6 = np.sqrt(np.diag(pcov_6))
param_m_6 = ufloat(popt_6[0], error_6[0])
param_b_6 = ufloat(popt_6[1], error_6[1])
print("Ausgleichsgerade VI: m, b:", param_m_6, param_b_6) if PRINT else print()

## Eintragen der Messwerte
plt.errorbar(noms(1/U_B_err), noms(Empf_err),
             xerr=stds(1/U_B_err), yerr=stds(Empf_err),
             fmt="rx", label="Messwerte")

## Regressionsgerade
X = np.linspace(1.5 * 1e-03, 6.0 * 1e-03, 300)
plt.plot(X, gerade(X, *popt_6), color="gray", label="Regressionsgerade")

## Ploteinstellungen
plt.grid()
plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _:
    float(x * 1e3)))
plt.xlabel(r"Reziproke Beschleunigungsspannung $U_{B}^{-1} \ [\mathrm{mV^{-1}}]$",
           fontsize=14, family="serif")
plt.ylabel(r"Empfindlichkeit $\frac{D}{U_{d}} \ [\mathrm{\frac{cm}{V}}]$",
           fontsize=14, family="serif")
plt.legend(loc="best")
plt.tight_layout()
#plt.show()
plt.savefig("Grafiken/Messreihe_VI.pdf")
plt.clf()


# Laden der Kathodenstrahlröhren Daten
L, P, d_1, d_2 = np.loadtxt("Messdaten/Kathodendaten.txt")

# Berechnung des mittleren Abstandes

d = (d_1 + d_2)/2
d_kplx = (d_1 + (d_1 + d_2)/2)/2
d_kplx_kplx = ((d_1 * 0.55) + (((d_1 + d_2)/2) * 0.45))
print(d_kplx, d_kplx_kplx)

# Berechnung des theoretischen Vergleichswerts
theo = L * P /(2 * d)
theo_kplx = L * P /(2 * d_kplx)
theo_kplx_kplx = L * P /(2 * d_kplx_kplx)
print("Theoriewert:", theo, theo_kplx, theo_kplx_kplx)
print("Abweichung vom Theoriewert:", np.abs(param_m_6 - theo_kplx_kplx)/theo_kplx_kplx)


# Vergleich des Theoriewerts mit dem berechnten
diff = np.abs(param_m_6 - theo_kplx)
diff_rel = diff / theo_kplx
diff_kplx = np.abs(param_m_6 - theo_kplx_kplx)
diff_rel_kplx = diff_kplx / theo_kplx_kplx
print("Unterschied der Ergebnisse:", diff, diff_rel, diff_kplx, diff_rel_kplx)


#==============================================================================
#
# Oszilloskop
#
#==============================================================================
#Laden der Frequenz und Verhältnisse
f_sz, n = np.loadtxt("Messdaten/Oszilloskop.txt", unpack=True)

# Laden des Frequenzfehlers
f_err = np.loadtxt("Messdaten/Fehler_Saegezahn.txt")

# Berechnung der fehlerbehafteten Größe
f_sz_err = unp.uarray(f_sz, [f_err]*len(f_sz))

#Berechnung der Sinusfrequenz
f_sin_err = f_sz_err * n
print("Sinusspannung:", f_sin_err[0], f_sin_err[1], f_sin_err[2], f_sin_err[3])

# Erstellen der Tabelle
T_osz = Table(siunitx=True)
T_osz.label("Auswertung_Oszilloskop")
T_osz.caption("Frequenzen $f_{sz}$ und $f_{sin}$ für stehende Wellen")
T_osz.layout(seperator="column", title_row_seperator="double", border=True)
T_osz.addColumn(f_sz_err, title="Sägezahn-Frequenz",
                symbol="f_{sz}", unit="\hertz")
T_osz.addColumn(n, title="Verhältnis", symbol="n")
T_osz.addColumn(f_sin_err, title="Sinus-Frequenz",
                symbol="f_{sin}", unit="\hertz")
#T_osz.save("Tabellen/Oszilloskop.tex")
#T_osz.show()

#==============================================================================
#
# Elektronen im magenetischen Feld
#
#==============================================================================

# Laden der Spulendaten
R_sp, N_sp = np.loadtxt("Messdaten/Spulendaten.txt")


# Berechnung des Magnetfeldes
def magnetfeld(I):
    global R_sp, N_sp
    return const.mu_0 * (8/m.sqrt(125)) * (N_sp * I/R_sp)

# Fehler des Magnetfeldes
M, N, I, R = sp.var("M N I R")
func_B = M * 8/sp.sqrt(125) * N * I/R
error_B = ErrorEquation(func_B, name="B_d", err_vars=[I])
print("B Fehler:", error_B.std)

# Laden der Ablenkströme
I_D_1, I_D_2, I_D_3, I_D_4 = np.loadtxt("Messdaten/Ablenkstrom.txt",
                                        unpack=True)

I_D_2 = np.delete(I_D_2, (np.where(I_D_2 == -100)[0]))
I_D_3 = np.delete(I_D_3, (np.where(I_D_3 == -100)[0]))
I_D_4 = np.delete(I_D_4, (np.where(I_D_4 == -100)[0]))


# Laden des Stromfehlers
i_d_err = np.loadtxt("Messdaten/Fehler_Ablenkstrom.txt")

# Erstellen der fehlerbehafteten Größe
I_D_1_err = unp.uarray(I_D_1, [i_d_err]*len(I_D_1))
I_D_2_err = unp.uarray(I_D_2, [i_d_err]*len(I_D_2))
I_D_3_err = unp.uarray(I_D_3, [i_d_err]*len(I_D_3))
I_D_4_err = unp.uarray(I_D_4, [i_d_err]*len(I_D_4))

# Berechnung der Ablenkmagnetfelder
B_D_1_err = magnetfeld(I_D_1_err)
B_D_2_err = magnetfeld(I_D_2_err)
B_D_3_err = magnetfeld(I_D_3_err)
B_D_4_err = magnetfeld(I_D_4_err)


# Berechnung der Distanzen
D = np.arange(0, 8, 1) * (const.inch/4) * 1e02  # cm

#Fehler der Distanzen
d_err = 0.2 * (const.inch/4) * 1e02  # cm

# Berechnung der fehlerbehafteten Größen
D_err = unp.uarray(D, [d_err]*len(D))


# Bestimmung des Quotienten D/(L² + D²)
Quot_err = D_err/(L**2 + D_err**2)

# Erstellen der Tabelle
T_Mag = Table(siunitx=True)
T_Mag.caption("Messdaten zur Bestimmung des Zusammenhangs zwischen $I_d$ und $D$")
T_Mag.label("Auswertung_Messdaten_II")
T_Mag.layout(seperator="column", border=True)
T_Mag.addColumn(I_D_1_err, title="1", symbol="")
T_Mag.addColumn(np.concatenate((I_D_2_err, [0])), title="2", symbol="")
T_Mag.addColumn(np.concatenate((I_D_3_err, [0,0])), title="3", symbol="")
T_Mag.addColumn(np.concatenate((I_D_4_err, [0,0])), title="4", symbol="")
T_Mag.addColumn(D_err, title="Verschiebung", symbol="D", unit="\centi\meter")
#T_Mag.show()
#T_Mag.save("Tabellen/Messwerte_II.tex")
T_Mag_2 = Table(siunitx=True)
T_Mag_2.caption("Messdaten zur Bestimmung des Zusammenhangs zwischen $B_d$ und $D$")
T_Mag_2.label("Auswertung_Messdaten_II_B")
T_Mag_2.layout(seperator="column", border=True)
T_Mag_2.addColumn(B_D_1_err, title="1", symbol="")
T_Mag_2.addColumn(np.concatenate((B_D_2_err, [0])), title="2", symbol="")
T_Mag_2.addColumn(np.concatenate((B_D_3_err, [0,0])), title="3", symbol="")
T_Mag_2.addColumn(np.concatenate((B_D_4_err, [0,0])), title="4", symbol="")
T_Mag_2.addColumn(D_err, title="Verschiebung", symbol="D", unit="\centi\meter")
#T_Mag_2.show()
#T_Mag_2.save("Tabellen/Messwerte_II_B.tex")



# Erstellen des Plots


#==============================================================================
class MessreiheI:
    pass
#==============================================================================

## Berechnung der Ausgleichsgerade
popt_7, pcov_7 = curve_fit(gerade, noms(B_D_1_err), noms(Quot_err),
                           sigma=stds(B_D_1_err))
error_7 = np.sqrt(np.diag(pcov_7))
param_m_7 = ufloat(popt_7[0], error_7[0])
param_b_7 = ufloat(popt_7[1], error_7[1])

print("Ausgleichsgerade VII:", param_m_7, param_b_7)


## Plot der Messwerte
plt.errorbar(noms(B_D_1_err), noms(Quot_err),
             xerr=stds(B_D_1_err), yerr=stds(Quot_err),
             fmt="rx", label="Messwerte")

## Plot der Regressionsgerade
X = np.linspace(-2e-05, 16e-05, 100)
#X = np.linspace(-2e-03, 16e-03, 100)
plt.plot(X, gerade(X, *popt_7), color="gray", label="Regressionsgerade")

## Ploteinstellungen
plt.grid()
plt.gca().yaxis.set_major_formatter(mpl.ticker.FuncFormatter
                                    (lambda x, _: float(x * 1e02)))  # 1/m
plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter
                                    (lambda x, _: float(x * 1e03)))  # µm
#plt.xlim(-40, 25)
#plt.ylim(-1, 6)
plt.ylabel(r"$\frac{D}{L^{2} + D^{2}} \ [\mathrm{m^{-1}}]$",
           fontsize=14, family='serif')
plt.xlabel(r"Magnetfeld $B_d \ [\mathrm{mT}]$",
           fontsize=14, family='serif')
plt.legend(loc="best")
plt.tight_layout()
#plt.show()
plt.savefig("Grafiken/BFeld_Messreihe_I.pdf")
plt.clf()

#==============================================================================
class MessreiheII:
    pass
#==============================================================================

## Berechnung der Ausgleichsgerade
popt_8, pcov_8 = curve_fit(gerade, noms(B_D_2_err), noms(Quot_err[:-1]),
                           sigma=stds(Quot_err[:-1]))
error_8 = np.sqrt(np.diag(pcov_8))
param_m_8 = ufloat(popt_8[0], error_8[0])
param_b_8 = ufloat(popt_8[1], error_8[1])

print("Ausgleichsgerade VIII:", param_m_8, param_b_8)

## Plot der Messwerte
plt.errorbar(noms(B_D_2_err), noms(Quot_err[:-1]),
             xerr=stds(B_D_2_err), yerr=stds(Quot_err[:-1]),
             fmt="rx", label="Messwerte")

## Plot der Regressionsgerade
X = np.linspace(-2e-05, 16e-05, 100)
plt.plot(X, gerade(X, *popt_8), color="gray", label="Regressionsgerade")

## Ploteinstellungen
plt.grid()
plt.gca().yaxis.set_major_formatter(mpl.ticker.FuncFormatter
                                    (lambda x, _: float(x * 1e02)))  # 1/m
plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter
                                    (lambda x, _: float(x * 1e03)))  # µm
#plt.xlim(-40, 25)
#plt.ylim(-1, 6)
plt.ylabel(r"$\frac{D}{L^{2} + D^{2}} \ [\mathrm{m^{-1}}]$",
           fontsize=14, family='serif')
plt.xlabel(r"Magnetfeld $B_d  \ [\mathrm{mT}]$",
           fontsize=14, family='serif')
plt.legend(loc="best")
plt.tight_layout()
#plt.show()
plt.savefig("Grafiken/BFeld_Messreihe_II.pdf")
plt.clf()

#==============================================================================
class MessreiheIII:
    pass
#==============================================================================

## Berechnung der Ausgleichsgerade
popt_9, pcov_9 = curve_fit(gerade, noms(B_D_3_err), noms(Quot_err[:-2]),
                           sigma=stds(Quot_err[:-2]))
error_9 = np.sqrt(np.diag(pcov_9))
param_m_9 = ufloat(popt_9[0], error_9[0])
param_b_9 = ufloat(popt_9[1], error_9[1])

print("Ausgleichsgerade IX:", param_m_9, param_b_9)

## Plot der Messwerte
plt.errorbar(noms(B_D_3_err), noms(Quot_err[:-2]),
             xerr=stds(B_D_3_err), yerr=stds(Quot_err[:-2]),
             fmt="rx", label="Messwerte")

## Plot der Regressionsgerade
X = np.linspace(-2e-05, 16e-05, 100)
plt.plot(X, gerade(X, *popt_9), color="gray", label="Regressionsgerade")

## Ploteinstellungen
plt.grid()
plt.gca().yaxis.set_major_formatter(mpl.ticker.FuncFormatter
                                    (lambda x, _: float(x * 1e02)))  # 1/m
plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter
                                    (lambda x, _: float(x * 1e03)))  # µm
#plt.xlim(-40, 25)
#plt.ylim(-1, 6)
plt.ylabel(r"$\frac{D}{L^{2} + D^{2}} \ [\mathrm{m^{-1}}]$",
           fontsize=14, family='serif')
plt.xlabel(r"Magnetfeld $B_d  \ [\mathrm{mT}]$",
           fontsize=14, family='serif')
plt.legend(loc="best")
plt.tight_layout()
#plt.show()
plt.savefig("Grafiken/BFeld_Messreihe_III.pdf")
plt.clf()

#==============================================================================
class MessreiheIV:
    pass
#==============================================================================

## Berechnung der Ausgleichsgerade
popt_10, pcov_10 = curve_fit(gerade, noms(B_D_4_err), noms(Quot_err[:-2]),
                             sigma=stds(Quot_err[:-2]))
error_10 = np.sqrt(np.diag(pcov_10))
param_m_10 = ufloat(popt_10[0], error_10[0])
param_b_10 = ufloat(popt_10[1], error_10[1])

print("Ausgleichsgerade X:", param_m_10, param_b_10)

## Plot der Messwerte
plt.errorbar(noms(B_D_4_err), noms(Quot_err[:-2]),
             xerr=stds(B_D_4_err), yerr=stds(Quot_err[:-2]),
             fmt="rx", label="Messwerte")

## Plot der Regressionsgerade
X = np.linspace(-2e-05, 16e-05, 100)
plt.plot(X, gerade(X, *popt_10), color="gray", label="Regressionsgerade")

## Ploteinstellungen
plt.grid()
plt.gca().yaxis.set_major_formatter(mpl.ticker.FuncFormatter
                                    (lambda x, _: float(x * 1e02)))  # 1/m
plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter
                                    (lambda x, _: float(x * 1e03)))  # µm
#plt.xlim(-40, 25)
#plt.ylim(-1, 6)
plt.ylabel(r"$\frac{D}{L^{2} + D^{2}} \ [\mathrm{m^{-1}}]$",
           fontsize=14, family='serif')
plt.xlabel(r"Magnetfeld $B_d  \ [\mathrm{mT}]$",
           fontsize=14, family='serif')
plt.legend(loc="best")
plt.tight_layout()
#plt.show()
plt.savefig("Grafiken/BFeld_Messreihe_IV.pdf")
plt.clf()

# Tabelle der Regeressionsparameter

## Speichern der Empfindlichleiten
param_B_m_err = np.array([param_m_7, param_m_8,
                     param_m_9, param_m_10])
## Speichen der Abschnitte
param_B_b_err = np.array([param_b_7, param_b_8,
                        param_b_9, param_b_10])




# Laden der Beschleunigungsspannung
U_B = np.loadtxt("Messdaten/Beschleunigungsspannung_B.txt", unpack=True)

# Laden des Fehlers
u_b_err = np.loadtxt("Messdaten/Fehler_Beschleunigungsspannung.txt")

# Erstellen der fehlerbehafteten Größen
U_B_err = unp.uarray(U_B, len(U_B)*[u_b_err])


# Berechnung der spezifischen Ladung
def spezLadung(m, UB):
    return 8 * m**2 * UB

# Fehlergleichung
U = sp.var("U_b")
func_Q = 8 * M**2 * U
error_Q = ErrorEquation(func_Q)
print("Fehler spez. Ladung:", error_Q.std)


Q_spez_1_err = spezLadung(param_m_7, U_B_err[0]) * 1e04
Q_spez_2_err = spezLadung(param_m_8, U_B_err[1]) * 1e04
Q_spez_3_err = spezLadung(param_m_9, U_B_err[2]) * 1e04
Q_spez_4_err = spezLadung(param_m_10, U_B_err[3]) * 1e04

Q_spez_err = np.array([Q_spez_1_err, Q_spez_2_err, Q_spez_3_err, Q_spez_4_err])
print("Spezifische Ladung:", Q_spez_1_err, Q_spez_2_err,
      Q_spez_3_err, Q_spez_4_err)

Q_spez_theo = 1.7588e11
print("Abweichung vom Lit-wert:", np.abs(Q_spez_err - Q_spez_theo)/Q_spez_err)

# Speichern der Fitparamter in Tabelle
T_params_B = Table(siunitx=True)
T_params_B.label("Auswertung_Parameter_E")
T_params_B.caption("Fit-Parameter der Daten aus den vier Messreihen")
T_params_B.layout(seperator="column", title_row_seperator="double",
                  border=True)
T_params_B.addColumn(param_B_m_err, title="Steigung",
                     symbol=r"\gamma", unit=r"\meter\per\volt")
T_params_B.addColumn(param_B_b_err, title="y-Achsenabschnitt",
                     symbol=r"\delta", unit=r"\per\centi\meter")
T_params_B.addColumn(Q_spez_err, title="spezifische Ladung",
                     symbol=r"\frac{e_0}{m_e}", unit=r"\coulomb\per\kilo\gram")
#T_params_B.show()
#T_params_B.save("Tabellen/Parameter_B.tex")
#==============================================================================
#
# Erdmagnetfeld
#
#==============================================================================

# Laden der Spannung, Stroms, Winkel
U_B_sp, I_hor, phi = np.loadtxt("Messdaten/Erdmagnetfeld.txt", unpack=True)

# Fehlerbehaftete Größe
I_hor_err = ufloat(I_hor, i_d_err)

# Berechnung des horizontalen Magnetfelds
B_hor_err = magnetfeld(I_hor_err)

# Berechnung des Totalen Magnetfeldes
B_tot_err = B_hor_err / m.cos(np.deg2rad(phi))

print("Horizontaler Strom I:", I_hor_err)
print("Horizontal Intensität B:", B_hor_err)
print("Totale Intensität B:", B_tot_err)









## Print Funktionen