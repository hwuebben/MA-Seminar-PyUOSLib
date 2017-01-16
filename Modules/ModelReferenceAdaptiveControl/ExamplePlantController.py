# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:30:30 2016

@author: joschnei
"""

from __future__ import division

import numpy as np

from BasicProcessingModule import BasicProcessingModule
from CPA import CPA

class ExamplePlantController(BasicProcessingModule):
    def __init__(self, foot):
        self.cpa = CPA(8,0,[-2, -2, -10, -6, -6],[2, 2, 10, 6, 6])
        BasicProcessingModule.__init__(self,foot)
        self.uc = np.zeros(2)
        self.yp = np.zeros(2)        
        self.output = 0
    
    def __call__(self, uc, yp, ym, u, index=0):
        self.uc[0] = self.uc[1]
        self.uc[1] = uc
        self.yp[0] = self.yp[1]
        self.yp[1] = yp
        
        inputVector = np.concatenate((self.uc, [u], self.yp))

        result = self.cpa.getPoint(inputVector)   

        error = ym-yp        
        
        toLearn = result + 0.5*error
             
        self.cpa.learn(inputVector, toLearn, l=1)
        
        return result
        
