# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
C:\Users\Josh\.spyder2\.temp.py
"""
from __future__ import (print_function)
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def eta(T,A,B):
    return A * np.exp(B/T)
    

t, v = np.loadtxt("Tabelle_T_eta.txt", unpack = True)
t_abs = t + 273
t_r = t_abs ** -1
popt, pcov = curve_fit(eta,t_abs,v)
error = np.sqrt(np.diag(pcov))
print("A= "+ str(popt[0])+"+/-"+str(error[0]) +" und B= "+str(popt[1])+"+/-"+str(error[1]))

plt.plot(popt[0],"ko",label = "A= "+ str(popt[0]))
plt.plot(popt[0],"ko",label = "B= "+ str(popt[1]))

plt.xlim(.0025,.004)
plt.ylim(1e-4,1e-2)

plt.yscale("log")
x = np.linspace(1e-6,1,1000)

plt.plot(t_r,v,"rx",label="Tabellenwerte")
plt.plot(x,eta(1/x,*popt),"b-",label="Fitkurve")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("fit.pdf")


