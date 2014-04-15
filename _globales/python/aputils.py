# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 23:30:43 2014

@author: Josh
"""

### From aputils.utils
from __future__ import (print_function,
                        division,
                        unicode_literals,
                        absolute_import)

import math
import numpy as np
import uncertainties as unc
import uncertainties.unumpy as unp
from sympy import *

from __future__ import (print_function,
                        division,
                        unicode_literals,
                        absolute_import)

import math
import numpy as np
import uncertainties as unc
import uncertainties.unumpy as unp
from sympy import *


class Quantity:
    def __init__(self, list_, err=None, factor=1):
        self.avr = np.mean(list_)*factor
        self.std = np.std(list_)*factor
        self.len = len(list_)
        self.std_err = self.std / math.sqrt(self.len)
        self.avr_err = unc.ufloat(self.avr, self.std_err)*factor

        if not err is None:
            Umean = unc.wrap(np.mean)
            self.list_err = unp.uarray(list_, [err]*self.len)*factor
            self.avr_err_gauss = Umean(self.list_err)*factor


class ErrorEquation:
    def __init__(f, symbol="", err_vars=None):
        self.var_equation = 0
        self.latex_names = dict()
        self.err_vars = err_vars
        self.function = f

        if self.err_vars is None:
            self.err_vars = function.free_symbols

        for v in self.err_vars:
            err = Symbol('latex_std_' + v.name)
            self.error_equation += self.function.diff(v)**2 * err**2
            self.latex_names[err] = '\\sigma_{' + latex(v) + '}'

        self.std = ('\\sigma_{' + symbol + '}=' +
                    latex(sqrt(s), symbol_names=latex_names))

    def show():
        pass


def error(f, symbol="", err_vars=None):
    s = 0
    latex_names = dict()

    if err_vars is None:
        err_vars = f.free_symbols

    for v in err_vars:
        err = Symbol('latex_std_' + v.name)
        s += f.diff(v)**2 * err**2
        latex_names[err] = '\\sigma_{' + latex(v) + '}'
    return ('\\sigma_{' + symbol + '}=' +
            latex(sqrt(s), symbol_names=latex_names))

### From aputils.latextables.tables
from string import join
from uncertainties import UFloat
from string import Formatter



class TableCell():
    def __init__(self, content=None):
        self.content = content
        self.fmtr = Formatter()
    def __repr__(self):
        return self.display()
    def display(self):
        """ type dependent string formatting """
        if isinstance(self.content, UFloat):
            return "{}".format(self.fmtr.format("{0:.1uS}", self.content))
        elif isinstance(self.content, int):
            return "{}".format(self.fmtr.format("{0:.0f}", self.content))
        elif isinstance(self.content, float):
            return "{}".format(self.fmtr.format("{0:.3f}", self.content))
        elif isinstance(self.content, basestring):
            return self.content
        elif self.content is None:
            return "None"
        else:
            return str(self.content)


class TableElement(list):
    def __init__(self, *elements):
        if elements:
            self.element = []
            for elem in elements:
                if isinstance(elem, tuple) or isinstance(elem, list):
                    for subelem in elem:
                        self.element.append(TableCell(subelem))
                else:
                    cellcontent = TableCell(elem)
                    self.element.append(cellcontent)
        else:
            self.element = [TableCell()]

    def next(self):
         if not self.element:
             raise StopIteration
         return self.element.pop()

    def __getitem__(self, index):
        return self.element[index]

    def __setitem__(self, key, value):
        self.element[key] = TableCell(value)

    """multiplication with an integer changes
    the length of Element by this factor"""
    def __mul__(self, other):
        return self.__class__(self.element * other)

    def __repr__(self):
        return str(self.element)

    def append(self, other):
        self.element.append(other)

    def display(self):
        assert False, "the method display has to be defined by subclasses"


class TableRow(TableElement):

    def display(self):
        return(join(map(lambda x: str(x), self.element)))


class TableColumn(TableElement):
    def display(self):
        return(join(map(lambda x: str(x) + "\n", self.element), sep=""))


class Table:
    def __init__(self, siunitx=False):
        self._titlerows = []
        self._titlecols = []
        self._rows = []
        self._columns = []
        self._rowcount = 0
        self._columncount = 0
        self.rowseperator = False
        self.titlerowseperator = "single"
        self.border = False  # either True or False
        self.centered = True  # either True or False
        self.position = "!h"    # h,b,t with or without !
        self.globalalign = ""
        self.columnseparator = False
        self._columnalignment = []
        self.labelprefix = "tab:"
        self.label = ""
        self.caption = ""
        self._tablepath = ""
        self._siunitx = siunitx
# TODO: possibility to add colums and rows of differnet length

    def addColumn(self, column, align="c", title=None, symbol=None, unit=None):
        # processing of the column title row
        if not unit is None and not symbol is None:
            symbol = symbol + (" [\\si{{{}}}]" if self._siunitx
                               else " [{}]").format(unit)
        titlecolumn = [title, symbol]
        if not self._titlerows:
            for item in titlecolumn:
                if not item is None:
                    self._titlerows.append(TableRow(item))
        else:
            for (trow, item) in zip(self._titlerows, titlecolumn):
                if not item is None:
                    trow.append(TableCell(item))

        # processing of the table columns
        self._columnalignment.append(align)
        self._columns.append(TableColumn(column))
        self._columncount += 1

        # processing of the table rows
        if not self._rows:
            for item in column:
                self._rows.append(TableRow(item))
            self._rowcount += len(column)
        else:
            for (row, item) in zip(self._rows, column):
                row.append(TableCell(item))

    def addRow(self, row):
        self._rows.append(TableRow(row))
        self._rowcount += 1
        if not self._columns:
            for item in row:
                self._columns.append(TableColumn(item))
            self._columncount += len(row)
        else:
            for (col, item) in zip(self._columns, row):
                col.append(TableCell(item))

    def layout(self, seperator="none", titlerowseperator="single",
               border=False):
        """
        Method to set the wanted seperators and border layout:

            - seperator = "none"|"row"|"column"|"both"
            - titlerowseperator = "single"|"double"|"none"
            - border = False|True
        """
        if seperator == "none":
            self.rowseperator = self.columnseparator = False
        elif seperator == "both":
            self.rowseperator = self.columnseparator = True
        elif seperator == "row":
            self.rowseperator = True
            self.columnseparator = False
        elif seperator == "column":
            self.rowseperator = False
            self.columnseparator = True
        self.titlerowseperator = titlerowseperator
        self.border = border



    def _latexColumns(self):
        columnsettings = "|{}|" if self.border else "{}"
        seperator = "|" if self.columnseparator else ""
        columnsettings = columnsettings.format(join(self._columnalignment,
                                                    sep=seperator))
        return columnsettings

    def _rowEnd(self):
        return "\\\\\hline" if self.rowseperator else "\\\\"

    def _latexTitleRows(self):
        trows = []
        if self._titlerows:
            for trow in self._titlerows:
                trows.append(join(map(lambda s: str(s), trow.element),
                            sep=" & ") + "\\\\" + "\n")
        return trows

    def _latexRows(self):
        rows = []
        for row in self._rows:
            if self._siunitx:
                rows.append(join(map(lambda s: "\\num{{{}}}".format(str(s)),
                                     row.element), sep=" & ")
                            + self._rowEnd() + "\n")
            else:
                rows.append(join(map(lambda s: str(s), row.element), sep=" & ")
                            + self._rowEnd() + "\n")

        if self.rowseperator:
            rows[-1] = rows[-1].replace("\\hline", "")
        return rows

    def _latexTable(self, path):
        tablebegin = "\\begin{table}" + "[{}]\n".format(self.position)
        tableend = "\\end{table}\n"
        tabularbegin = "\t\\begin{tabular}" + \
                       "{{{}}}\n".format(self._latexColumns())
        tabularend = "\t\\end{tabular}\n"
        label = "\\label{{{}}}".format(self.labelprefix + self.label)
        caption = "\t\\caption{{{} {}}}\n".format(self.caption, label)
#        print(tablebegin, tableend, tabularbegin, tabularend, label, caption)
        texfile = open(path, "w")
        texfile.write(tablebegin + ("\t\\centering\n" if self.centered else "")
                      + tabularbegin)
        if self.border:
            texfile.write("\t\t\\hline\n")
        if self._titlerows:
            for line in self._latexTitleRows():
                texfile.write("\t\t" + line)
            if self.titlerowseperator == "single":
                texfile.write("\\hline")
                texfile.write("\n")
            elif self.titlerowseperator == "double":
                texfile.write("\\hline")
                texfile.write("\\hline")
                texfile.write("\n")

        for line in self._latexRows():
            texfile.write("\t\t" + line)
        if self.border:
            texfile.write("\t\t\\hline\n")
        texfile.write(tabularend + caption + tableend)
        texfile.close()

    def save(self, tablepath, temp=False):
        #TODO: If reset the tablepath when in TEMP dir
        if not temp:
            self._tablepath = tablepath
            self._latexTable(self._tablepath)
        self._latexTable(tablepath)

    def show(self):
        if not self._tablepath:
            conv = Converter(tableobj=self)
            _show(conv.texToPng(), conv)
        else:
            conv = Converter(self._tablepath)
            _show(conv.texToPng(), conv)

    def __repr__(self):
        return join(map(lambda row: row.display(), self._rows), sep="\n")

### From aputils.latextables.preview.core
import os


class _ChDir():
    """wrapper class for directory changes"""

    def __init__(self, path):
        self.lastpath = os.getcwd()
        os.chdir(path)

    def __del__(self):
        os.chdir(self.lastpath)

### From aputils.latextables.preview.conversion

import tempfile
import shutil
import datetime


class Converter:

    def __init__(self, tablepath=None, tableobj=None):
        self._id = self._calcId()
        self._preamble = "\\documentclass{article}\n " + \
                         "\\usepackage{siunitx}\n" + \
                         "\\sisetup{locale = DE}\n" +\
                         "\\sisetup{prefixes-as-symbols = false}\n" + \
                         "\\sisetup{separate-uncertainty = true}\n" +\
                         "\\pagestyle{empty}\n"
        self._docbegin = "\\begin{document} \n"
        self._docend = "\\end{document}"
        self._temppath = tempfile.gettempdir()
        self._tempfoldername = "latextables"
        self._tableobj = tableobj
        if tablepath is None and not tableobj is None:
            self._tablepath = self._temppath + "\\" + self._tempfoldername\
                               + "\\" + self._id + "\\" + "table.tex"
        else:
            self._tablepath = os.path.abspath(tablepath)

    def _calcId(self):
        date = str(datetime.datetime.now().date())
        date = date.replace("-", "")
        time = str(datetime.datetime.now().time())
        dotpos = time.rfind(".")
        time = time[:dotpos].replace(":", "")
        return date + time

    def _tempTexFile(self):
        """generating the temporary tex file"""
        if not self._tempfoldername in os.listdir(self._temppath):
            os.mkdir(self._temppath + "\\" + self._tempfoldername)

        if not self._id in os.listdir(self._temppath + "\\" +
                                           self._tempfoldername):
            os.mkdir(self._temppath + "\\" +
                     self._tempfoldername + "\\" + self._id)

        if not self._tableobj is None:
            self._tableobj.save(self._tablepath, temp=True)

        texfile = open(self._temppath + "\\" + self._tempfoldername +
                       "\\" + self._id + "\\" + "temp.tex", "w")
        tablelines = open(self._tablepath, "r").readlines()
        table = join(tablelines)
        texfile.write(self._preamble + self._docbegin + table + self._docend)
        texfile.close()

    def _dviToPngFile(self):
        """generating the temporary dvi file and
           converting it to the png file"""
        cdir = _ChDir(self._temppath + "\\" +
                      self._tempfoldername + "\\" + self._id)
        os.system("latex {}".format("temp.tex"))
        os.system("dvipng -T tight -D 240 -o temp.png {}".format("temp"))
        del cdir

    def deleteTemp(self):
        shutil.rmtree(self._temppath + "\\" + self._tempfoldername)

    def texToPng(self):
        self._tempTexFile()
        self._dviToPngFile()
        return self._temppath + "\\" + self._tempfoldername \
               + "\\" + self._id + "\\" + "temp.png"

### From aputils.latextables.preview.gui
import sys

from PyQt4 import QtGui


class Previewer(QtGui.QWidget):

    def __init__(self, imgpath, obj):
        super(Previewer, self).__init__()
        self.initUI(imgpath)
        self.obj = obj

    def initUI(self, imgpath):
        hbox = QtGui.QHBoxLayout(self)
        pixmap = QtGui.QPixmap(imgpath)

        lbl = QtGui.QLabel(self)
        lbl.setPixmap(pixmap)

        hbox.addWidget(lbl)
        self.setLayout(hbox)

        self.move(300, 200)
        self.setWindowTitle('Table Preview')
        self.show()

    def closeEvent(self, event):
        conversion.Converter.deleteTemp(self.obj)


def _show(imgpath, obj):
    app = QtGui.QApplication(sys.argv)
    preview = Previewer(imgpath, obj)
    app.exec_()
