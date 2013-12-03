# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:28:37 2013

@author: JoshLaptop
"""

from __future__ import unicode_literals
import numpy as np
import string
import uncertainties as unc
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as n,
                                  std_devs as s)

# Class for shorthand formating uncertainties, eg. 5.6 +/- 0.4 => 5.6(4)
class ShorthandFormatter(string.Formatter):
    def format_field(self, value, format_spec):
        if isinstance(value, unc.UFloat):
            return value.format(format_spec + 'S')  # Shorthand option added
        # Special formatting for other types can be added here (floats, etc.)
        else:
            # Usual formatting:
            return super(ShorthandFormatter, self).format_field(value,
                                                                format_spec)
#Object of the ShorthandFormatter Class
fmtr = ShorthandFormatter()


def entryFmt(x):
    if isinstance(x, unc.UFloat):
        return fmtr.format("{0:.1u}", x)
    elif isinstance(x, float):
        return fmtr.format("{0:.3f}", x)



# TODO: implement function to allow multiple columns with the same
# TODO: header or title


# returns True if val is the last entry in the array arr
def _isLast(val, arr):
    return (np.where(arr == val)[0])[0] == (len(arr) - 1)


# retruns the next formating symbol, depending on whether val is the last
# entry of the current row or not
def _rowFmt(val, arr):
    if _isLast(val, arr):
        return r"\\"
    else:
        return " & "


def formatFmt(arr):
    frmt = "|"
    for i in arr:
        frmt += i
        frmt += "|"
    return frmt


## TODO: improve uncertainties
def toTable(cols, col_titles=None, col_syms=None,
            col_units=None, fmt=None, cap=None, label=None):
    # initialization of the variable containing the \begin statement of
    # the latex table-enviornment and the latex formatting of columns
    begin = (r"\begin{table}[!h]" + "\n\t" + r"\centering" "\n\t" +
             r"\begin{tabular}" + "{{{}}}".format(formatFmt(fmt))
             + "\n" + "\t\t" + r"\hline" + "\n")

    # initialization of the variable containing the \end statement of
    # the latex table-enviornment and the caption and label if provided
    end = ("\t\t" + r"\hline" + "\n\t" + r"\end{tabular}" + "\n\t"
           r"\caption{{{} \label{{tab:{}}}}}".format(cap, label) +
           "\n" + "\end{table}")

    # initialization of the variable for the titles of each column,
    # for instance the name of the displayed values
    titles = "\t\t"

    # col_titles, col_syms and col_units have to be numpyarrays
#    __temp = [col_titles, col_syms, col_units]
#    for i in __temp:
#        if not isinstance(i, np.ndarray):
#            i = np.array(i)
    col_titles = np.array(col_titles)
    col_syms = np.array(col_syms)
    col_units = np.array(col_units)
    if not col_titles is None:
        for t in col_titles:
            title = ("{}".format(t)) + _rowFmt(t, col_titles)
            titles += title
        titles += "\n"

    # initilisation of the variable containing the header for each column,
    # for instance the symbol and unit of measurement of the displayed values
    headers = "\t\t"
    if not col_syms is None:
        if not col_units is None:
            for i in range(len(col_syms)):
                header = (r"${}\,[\si{{{}}}]$".format(col_syms[i],
                          col_units[i]) + _rowFmt(col_syms[i], col_syms))
                headers += header
                if headers.endswith(r"\\"):
                    headers += r"\hline\hline" + "\n"
        else:
            for i in range(len(col_syms)):
                headers = ("{}".format(col_syms[i]))
                headers += header
                if headers.endswith(r"\\"):
                    headers += r"\hline\hline" + "\n"
    else:
        headers = ""

    # initialization of variables to format the rows of entries the right way
    row = ""
    rows = ""
    # formatting the rows
    if not cols is None:
        if not all(isinstance(i, np.ndarray) for i in cols):
            print "cols must to be an ndarray-Type"
        else:
            cols = np.transpose(cols)
            for k in cols:
                row = "\t\t"
                for i in k:
                    row += r"\num{{{}}} ".format(entryFmt(i)) + _rowFmt(i, k)
                row += "\n"
                rows += row

    return (begin + titles + headers + rows + end).encode("UTF-8")
