# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:30:30 2016

@author: joschnei
"""

from __future__ import division

import numpy as np

from BasicProcessingModule import BasicProcessingModule

class ExamplePlant(BasicProcessingModule):
    def __init__(self, foot):
	BasicProcessingModule.__init__(self,foot)
	self.state = [0,0,0]
        self.output = 0
        
    
    def __call__(self, u, index=0):
        ypk = self.state[0]
        ypkm = self.state[1]
        uk = u
        ukm = self.state[2]
        pi = np.pi
        
        #yp = 0.3*ypk + 0.6*ypkm + 0.6*np.sin(pi*uk) - 0.3*np.sin(3*pi*uk) - 0.2*ukm**3 + 0.4*ukm

        yp = (ypk)/(1+ypk**2+ypkm**2) + 1/(1+np.exp(-ypk-ypkm)) + uk + 0.4*ukm        
        
        self.state[1] = self.state[0]
        self.state[0] = yp
        
        return yp
        
