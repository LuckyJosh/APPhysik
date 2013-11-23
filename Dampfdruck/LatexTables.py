# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:28:37 2013

@author: JoshLaptop
"""
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
import numpy as np


def temp(a, b):
    if a == b[-1]:
        return r"\\"
    else:
        return " & "
        
def temp2(a, b):
    if a == b:
        return r"\\" + r" \hline\hline" + "\n"
    else:
        return " & "


def toTable(cols, col_titles=None, col_syms=None, col_units=None, cap=None, label=None):
    begin = (r"\begin{table}" + "\n\t" + r"\centering" "\n\t" +
             r"\begin{tabular}{}" + "\n" + "\t\t" + r"\hline" + "\n")
    header = "\t"     
    if not col_syms is None:
        if not col_units is None:
            for i in range(len(col_syms)):
                headers = (r"{}\,[\si{{{}}}]".format(col_syms[i], col_units[i])
                           + temp2(i ,(len(col_syms)-1)))
                header += headers 
        else:
            for i in range(len(col_titles)):
                headers = ("{}".format(col_syms[i]))
                header += headers
    else: 
        header = ""
    
    end = ("\t\t" + r"\hline" + "\n\t" + r"\end{tabular}" + "\n\t"
           r"\caption{{{} \label{{tab:{}}}}}".format(cap, label) +
           "\n" + "\end{table}")
    row = ""
    rows = ""
    if not cols is None:
        if not all(isinstance(i, np.ndarray) for i in cols):
            "cols must to be an ndarray-Type"
        else:
            cols = np.transpose(cols)
            for k in cols:
                row = "\t\t"
                for i in k:
                    row += r"\num{{{}({})}} ".format(noms(i), int(stds(i))) + temp(i, k)
                row += "\n"
                rows += row
    return begin + header + rows + end
