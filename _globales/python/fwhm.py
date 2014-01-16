# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 14:09:52 2013

@author: Josh
"""
from __future__ import (print_function,
                        division,
                        unicode_literals,
                        absolute_import)

import matplotlib.pyplot as plt
import numpy as np


def FWHM(x, y, plot=False, col="gray"):
    hm = max(y)/2 
    d = np.sign(hm - y[0:-1]) - np.sign(hm - y[1:])
    left_idx = find(d > 0)[0]
    right_idx = find(d < 0)[-1]
    fw = x[right_idx] - x[left_idx]
    if plot:
        X = np.linspace(x[left_idx], x[right_idx])
        plt.plot(X, len(X)*[hm], color=col)
        plot = plt.bar(x[left_idx], hm, width=fw, color=col)
        plt.text(x[left_idx]+ fw/2, hm*0.5, "FWHM = {0:.3f}".format(fw), 
                 ha='center', va='bottom')
    else:
        return fw   
    