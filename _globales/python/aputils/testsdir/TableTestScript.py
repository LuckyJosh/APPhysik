# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 16:22:55 2014

@author: Josh
"""

import os
from uncertainties import ufloat
import numpy as np
import sys

_MODULEPATH = "D:\\Eigene Dateien\\Dokumente\\Programmierung\\Python\\PythonRepo\\APutils\\aputils"

if not _MODULEPATH in sys.path:
    sys.path.append(_MODULEPATH)
from latextables.tables import Table

#    c = TableCell("Title")
#    print(c)
#    a = TableRow(1, 2.2, ufloat(4.3149, 0.03), 5.000234, 7.423)
#    b = TableColumn(1, 2.2, ufloat(4.3149, 0.2), 5.000234, 7.423)
#    d = TableRow("Test", "Test2", "Test3")
#    e = TableColumn("Test", "Test2", "Test3")
#    print(a); print(a[2])
#    print(b); print(b[4])
#    print(d); print(d[1])
#    print(e); print(e[2])
T = Table()
T.layout(seperator="both", border=False)
T.caption = "this is the first Table with the new module"
T.label = "theFirst"
T.addColumn([1, 2, 3, 4, 5])
T.addColumn([1, 2, 3, 4, 5])
T.addColumn([1, 2, 3, 4, 5])
T.addColumn([1, 2, 3, 4, 5])
T.addColumn([1, 2, 3, 4, 5, 6])
T.addRow([1, 2, 3, 4, 5])
T.addRow([1, 2, 3, 4, 5])
T.addRow([1, 2, 3, 4, 5])
T.addRow([1, 2, 3, 4, 5])
T2 = Table(siunitx=True)
T2.layout(seperator="column", titlerowseperator="none", border=True)
T2.caption = "this is the second Table with the new module"
T2.label = "theSecond"
T2.addColumn([ufloat(0.3423, 0.4), 2, 3, 4, 5], title="Test 1", symbol="A", unit="C")
T2.addColumn([ufloat(0.3423, 0.4),2, 3, 4, 5], title="Test 2", symbol="B" , unit="f")
T2.addColumn([ufloat(0.3423, 0.4), 2, 3, 4, 5], title="Test 3", symbol="c", unit="p")
T2.addColumn([ufloat(0.3423, 0.4), 2, 3, 4, 5], title="Test 4", symbol="D", unit = "q")
T2.addColumn([ufloat(0.3423, 0.4), 2, 3, 4, 5], title="Test 5", symbol="4", unit="o")
T2.addRow([1, 2, 3, 4, 5])
T2.addRow([1, 2, 3, 4, ufloat(3.038, 0.2894)])
T2.addRow([1, 2, 3, 4, 5])
T2.addRow([1, ufloat(1000, 23), 3, 4, 5])

#    print T._rows
#    print(T._columns)
#    for row in T._rows:
#        print(row.display())

T.save("latextables.tex")
#    T2.save("../latextables2.tex")
print(4 + 5)
