# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 05:55:18 2014

@author: JoshLaptop
"""
import os
from string import join
import tempfile
import shutil
import datetime
from core import _ChDir




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
            self._tableobj._tempsave(self._tablepath)

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
        os.system("latex {} > NUL".format("temp.tex"))
        os.system("dvipng -T tight -D 240 -o temp.png {} > NUL".format("temp"))
        del cdir

    def deleteTemp(self):
        shutil.rmtree(self._temppath + "\\" + self._tempfoldername)

    def texToPng(self):
        self._tempTexFile()
        self._dviToPngFile()
        return self._temppath + "\\" + self._tempfoldername \
               + "\\" + self._id + "\\" + "temp.png"
