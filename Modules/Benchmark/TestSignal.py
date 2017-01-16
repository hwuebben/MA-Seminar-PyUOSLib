# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:57:40 2015

@author: JSCHOENK
"""
from __future__ import division
import numpy as np


from MPyUOSLib import BasicProcessingModule

"""
This module groups different test signals for identifying dynamic systems
"""

class APRBS(BasicProcessingModule):
    """
    Amplitude modulated pseudo random signal
    """
    def __init__(self, foot):
        default = {"width":1,"domain":[0,1]}
        BasicProcessingModule.__init__(self, foot, default)        
        
    def __call__(self, index=0):
        if np.mod(index,self.width)==0:
            self.output = np.random.uniform(low=self.domain[0],high=self.domain[1])
        return self.output