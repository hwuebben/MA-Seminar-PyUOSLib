# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:30:30 2016

@author: joschnei
"""

from __future__ import division

import numpy as np

from BasicProcessingModule import BasicProcessingModule

class ControllerTDL(BasicProcessingModule):
    def __init__(self, foot):
	BasicProcessingModule.__init__(self,foot)
        self.output = 0
        self.delay = 0
        if(self.delay > 0):
            self.arr = np.zeros(self.delay)
    
    def __call__(self, u, index=0):
        if(self.delay == 0):
            return u
        yp = u
        self.arr[0:-1] = self.arr[1::]
        self.arr[-1] = yp
        return self.arr[0]
        
