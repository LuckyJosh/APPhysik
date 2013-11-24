# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:28:37 2013

@author: JoshLaptop
"""

import numpy as np
from uncertainties.unumpy import (nominal_values as n,
                                  std_devs as s)
def _aux(a, b):
    if a == b[-1]:
        return r"\\"
    else:
        return " & "
       
def _aux2(a, b):
    if a == b:
        return r"\\" + r" \hline\hline" + "\n"
    else:
        return " & "

#def _detStd(i):
#    for i in 

## TODO: improve uncertainties
def toTable(cols, col_titles=None, col_syms=None ,col_units=None, cap=None, label=None):
    begin = (r"\begin{table}" + "\n\t" + r"\centering" "\n\t" +
             r"\begin{tabular}{}" + "\n" + "\t\t" + r"\hline" + "\n")
    titles = "\t\t"
    headers = "\t\t"     
    if not col_titles is None:
        for t in col_titles: 
            title =("{}".format(t)) + _aux(t, col_titles) 
            titles += title
        titles += "\n"
    if not col_syms is None:
        if not col_units is None:
            for i in range(len(col_syms)):
                header = (r"${}\,[\si{{{}}}]$".format(col_syms[i], col_units[i])
                           + _aux2(i ,(len(col_syms)-1)))
                headers += header 
        else:
            for i in range(len(col_syms)):
                headers = ("{}".format(col_syms[i]))
                headers += header
    else: 
        headers = ""
    
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
                    row += r"\num{{{}({})}} ".format(n(i), s(i)) + _aux(i, k)
                row += "\n"
                rows += row
    return begin + titles + headers + rows + end

