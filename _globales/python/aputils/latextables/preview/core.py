# -*- coding: utf-8 -*-
"""
Created on Mon Apr 07 11:36:00 2014

@author: JoshLaptop
"""

import os

class _ChDir():
    """wrapper class for directory changes"""

    def __init__(self, path):
        self.lastpath = os.getcwd()
        os.chdir(path)

    def __del__(self):
        os.chdir(self.lastpath)
