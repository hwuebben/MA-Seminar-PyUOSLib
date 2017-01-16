# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:30:30 2016

@author: joschnei
"""

from __future__ import division

import numpy as np

from BasicProcessingModule import BasicProcessingModule

class LinearReferenceModel(BasicProcessingModule):
    def __init__(self, foot):
        BasicProcessingModule.__init__(self,foot)
        self.output = 0
        self.p = 2
        self.q = 2
        self.alpha = np.random.rand(self.p)
        self.beta = np.random.rand(self.q)
        self.uc = np.zeros(self.p)
        self.ym = np.zeros(self.q)

    def __call__(self, uc, index=0):
    
        #ym = np.dot(self.alpha, self.uc) + np.dot(self.beta, self.ym)
        ym = 0.6*self.ym[1] + 0.2*self.ym[0] + uc
    
        self.uc[0:-1] = self.uc[1::]
        self.uc[-1] = uc 
        self.ym[0:-1] = self.ym[1::]
        self.ym[-1] = ym

        return ym
