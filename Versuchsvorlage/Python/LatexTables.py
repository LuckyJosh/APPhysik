# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:28:37 2013

@author: JoshLaptop
"""

import numpy as np


def toTable(cap, label, cols=None):
    begin = (r"\begin{table}" + "\n\t" + r"\centering" "\n\t" +
             r"\begin{tabular}{}")
    end = ("\n\n\t" + r"\end{tabular}" + "\n\t"
           r"\caption{{{} \lable{{{}}}}}".format(cap, label) +
           "\n" + "\end{table}")
    rows_str = ""
    if not cols is None:
        if not all(isinstance(i, np.ndarray) for i in cols):
            "cols needs to be an ndarray-Type"
        else:
            rows = np.zeros(np.alen(cols))
            for i in range(np.alen(cols[0])):
                r = 0
                for j in cols:
                    rows[i] = j[i]
                    r += 1
            for k in rows        
                    rows_str += "{} & ".format(j)
                rows_str += r"\\" + "\n"

    return begin + rows_str + end

a = np.array([1, 2])
b = np.array([3, 4])

print(toTable("Test", "eq:test", [a, b]))
