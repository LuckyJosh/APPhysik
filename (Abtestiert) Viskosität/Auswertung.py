# -*- coding: utf-8 -*-
from __future__ import (print_function,
                        division,
                       unicode_literals,
                       absolute_import)

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const 
import numpy as np
import sympy as sp
from uncertainties import ufloat
import uncertainties.unumpy as unp


## Funktion für eta in Abhängikeit von Kugelkonstante "Gleichung(1)"
def eta_K(K,rho_k,rho_f,t):
    return K * (rho_k - rho_f )* t



## Definition der Andrade-Gleichung
def andrade(T,A,B):
    return A * np.exp(B / T) 


## Funktion zur Berechnung der der Reynoldzahl
def raynolds(d_k, v_k, rho_f, eta_f):
    return (rho_f * v_k * d_k)/eta_f

## Messwerte und gegebene Daten der Kugelgeometrie 
##(Masse, Durchmesser, Kugelkonstante)

    ## Kugel_kl
M_kl = ufloat(4.46e-3, 0.01e-3) #kg
D_kl = ufloat(1.56e-2,0.02e-2)  #m
K_kl = 0.07640e-6 #Pam**3/kg

    ## Kugel_gr
M_gr = ufloat(4.96e-3,0.01e-3)  #kg
D_gr = ufloat(1.58e-2, 0.02e-2) #m

## Messstrecke
X = 0.1 #m


## Dichten von Wasser bei unterschiedlichen Temperaturen
T_C, Rho_H2O = np.loadtxt("Literaturwerte/Wasserdichten.txt", unpack=True)




## Berechnete Kugelgeometrie (Radius, Volumen, Dichte)
    ## Kugel_kl
R_kl = D_kl / 2                   #m
V_kl = 4/3 * const.pi * (R_kl)**3 #m³
Rho_kl = M_kl / V_kl              #kg/m³
    ## Kugel_gr
R_gr = D_gr / 2                   #m
Rho_H2O_21 = 0.99799e3 # kg/m³
 
V_gr = 4/3 * const.pi * (R_gr)**3 #m³
Rho_gr = M_gr / V_gr              #kg/m³
uRho_gr = ufloat(unp.nominal_values(Rho_gr), unp.std_devs(Rho_gr))

print(R_kl)
print(R_gr)
print(V_kl)
print(V_gr)
print(Rho_kl)
print(Rho_gr)



## Berechnung der Mittelwerte der Fallzeiten für Kugel_gr und Kugel_kl
t_kl, t_gr = np.loadtxt("Messwerte/Zeitmessung_1.txt", unpack = True)
t_kl_avr = np.mean(t_kl)
t_kl_avr_err = np.std(t_kl) / np.sqrt(len(t_kl))
t_gr_avr = np.mean(t_gr) 
t_gr_avr_err = np.std(t_gr) / np.sqrt(len(t_gr))
print(t_kl_avr)
print(t_kl_avr_err)
print(t_gr_avr)
print(t_gr_avr_err)


## Berechnung der Zeitmittelwerte 
T_C,t_1,t_2,t_3,t_4 = np.loadtxt("Messwerte/Zeitmessung_2.txt", unpack = True)
t_avr = np.zeros(10)
t_avr_err = np.zeros(10)
for i in range(10):
    t_avr[i] = np.mean([t_1[i],t_2[i],t_3[i],t_4[i]])
    t_avr_err[i] = np.std([t_1[i],t_2[i],t_3[i],t_4[i]]) / np.sqrt(len([t_1[i],t_2[i],t_3[i],t_4[i]]))
ut_avr = unp.uarray(t_avr,t_avr_err)
print("Zeitfehler:")
print(t_avr_err)
## Berechnung der Kelvintemperaturen
T_K = T_C + 273.15


## Berechnung der Geschwindigkeit der großen Kugel
v_gr = X/ut_avr  
uv_gr = unp.uarray(unp.nominal_values(v_gr), unp.std_devs(v_gr)) 
print("Geschwindigkeiten:")
print(uv_gr)
   
## Berechnung der Viskosität bei 21°C aus Messwerten für Kugel_kl
eta_21 = eta_K(K_kl,Rho_kl,Rho_H2O[0],t_kl_avr)



## Berechnung der Kugelkonstante K_gr
K_gr = eta_21/(t_gr_avr*(Rho_gr - Rho_H2O[0]))
uK_gr = ufloat(unp.nominal_values(K_gr), unp.std_devs(K_gr))

print("uK_gr", K_gr)
print("Rho_gr", Rho_gr)
print("Rho_H2O", Rho_H2O)
print("t_avr", t_avr)

## Viskosität bei Temperatursteigerung
    ## eta_K Methode
eta_T_K = unp.uarray(np.zeros(10),np.zeros(10))
for i in range(10):
    eta_T_K[i] = eta_K(uK_gr,uRho_gr,Rho_H2O[i],ut_avr[i])
print(eta_T_K)

B = eta_K(uK_gr,uRho_gr,Rho_H2O[0],ut_avr[0])
print("Test",B)
for i in range(3):
    print(B.error_components().items()[i])





## Berechnung der Reynoldszahl Re_gr ~ 1e3

Re = unp.uarray(np.zeros(10),np.zeros(10))
for i  in range(10):
    Re[i] = raynolds(D_gr,uv_gr[i],Rho_H2O[i],eta_T_K[i])
print("Reynoldszahlen", Re)

## Erstellen der Plots
    ## Laden der Werte und Fehler     
eta = np.loadtxt("Ausgabe/Plot_Viskositaet.txt", unpack = True)
eta_err = np.loadtxt("Ausgabe/Plot_ViskositaetUnsicherheit.txt", unpack =  True)
    
    ## Erstellen der Fit-Kruve
popt,pcov = curve_fit(andrade,T_K,eta, sigma = eta_err)
fit_err = np.sqrt(np.diag(pcov))
print("A = "+ str(popt[0])+"+/-" + str(fit_err[0]) + " B = " + str(popt[1])+"+/-" + str(fit_err[1]))


    ## Plot mit linearen Achsen
plt.xlabel("$T\,[\mathrm{K}]$")
plt.ylabel(r"$\eta\,[10^{-3}\mathrm{Pa \cdot s}]$")
plt.xlim(T_K[0]-10,T_K[-1]+10)
plt.ylim(np.min(eta)-1e-04,np.max(eta)+1e-04)
plt.gca().yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: float(x * 1e3)))

T = np.linspace(273,400,1000)

plt.errorbar(T_K ,eta,yerr = eta_err ,fmt = "rx", label="Messwerte")

plt.plot(T,andrade(T,*popt), "b-",label="Fit-Kurve")
plt.tight_layout()
plt.legend(loc="best")
plt.savefig("Plots/Plot_eta_T.pdf")

plt.clf()


    ## Plot mit halblogarithmischer Skala und gegen 1/T
plt.xlabel(r"$\frac{1}{T}\,[10^{-2}\mathrm{\frac{1}{K}}]$")
plt.ylabel(r"$\eta\,[10^{-3}\mathrm{Pa \cdot s}]$")
#plt.xlim(1/T_K[-1]-1e-04,1/T_K[0] + 1e-04)
plt.xlim(0.24e-02, 1/T_K[0] + 1e-04)
plt.ylim(0.1e-03,3e-03)
plt.yscale("log")
plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x,_: float(x * 1e02)))
plt.gca().yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: float(x * 1e03)))
plt.grid(which="both")


plt.errorbar(1/T_K ,eta, yerr = eta_err, fmt ="rx", label="Messwerte")
plt.plot(1/T,andrade(T,*popt),"b-",label="Ausgleichs-Kurve")
plt.legend(loc="best")
plt.tight_layout()
plt.show()
#plt.savefig("Plots/Plot_eta_T_log.pdf")
print(1/T_K)



A_B = unp.uarray([popt[0],popt[1]],[fit_err[0],fit_err[1]])

    ## Schreiben der Fit-Parameter
np.savetxt("Ausgabe/AusgleichskurvenParameter.txt",A_B,fmt=str("%r"),header="A(oben) #B(unten)" )

    ## Schreiben dieser in 'Messwerte_Mittelwerte.txt'
T_values = np.array([T_C,t_1,t_2,t_3,t_4,t_avr])
T_values = T_values.T
np.savetxt("Ausgabe/Messwerte_Mittelwert.txt",T_values, fmt=str("%.2f"),
           header="T[°C] #t_1[s] #t_2[s] #t_3[s] #t_4[s] #t_avr[s]")








