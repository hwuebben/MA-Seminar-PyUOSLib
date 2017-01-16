# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:30:30 2016

@author: joschnei
"""

from __future__ import division

import numpy as np

from BasicProcessingModule import BasicProcessingModule

class ExternalInputUc(BasicProcessingModule):
    def __init__(self, foot):
        BasicProcessingModule.__init__(self,foot)
        self.output = 0
        self.step = 0
    
    def __call__(self, index=0):
        pi = np.pi
        k = self.step
        uc = np.sin((2*pi*k)/10)+np.sin((2*pi*k)/25)
        self.step += 1
        return uc
