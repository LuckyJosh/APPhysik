# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 12:29:48 2014

@author: Josh
"""

import os
import subprocess as sp
import sys

PATH = os.getcwd()

folders = [f for f in os.listdir(PATH) if os.path.isdir(f) and (not f[0] == "_"
           and not f == ".git")]

testfolders = [f for f in os.listdir(PATH) if "TestOrdner" in f]

class Make:
    def __init__(self, path):
        self.path = PATH + "\\" + path
        os.chdir(self.path)
    def clean(self):
        os.system("make clean")
        os.chdir(PATH)
    def make(self):
        os.system("make")
        os.chdir(PATH)


for folder in testfolders:
    make = Make(folder)
    make.clean()


os.system("git" + " add " + sys.argv[1])
os.system("git" + " commit -m " + sys.argv[2])
os.system("git" + " push")
