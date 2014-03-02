# -*- coding: utf-8 -*-
"""
Created on Sun Mar 02 17:18:40 2014

@author: Josh
"""

import os

L = [item[2:] for  item in os.listdir(os.getcwd()) if item[0]=="A"]
print(L)
