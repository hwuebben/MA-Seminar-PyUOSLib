# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:30:30 2016

@author: joschnei
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from BasicProcessingModule import BasicProcessingModule

class Summation(BasicProcessingModule):
    def __init__(self, foot):
        BasicProcessingModule.__init__(self,foot)
        self.output = 0
        self.cumLoss = [0]
        self.ym = []
        self.yp = []

    def __call__(self, ym, yp, index=0):
        self.ym.append(ym)
        self.yp.append(yp)
        self.cumLoss.append(self.cumLoss[-1] + np.abs(ym-yp))
        return self.cumLoss[-1]

    def end(self):
        plt.plot(self.ym)
        plt.plot(self.yp, 'r')
