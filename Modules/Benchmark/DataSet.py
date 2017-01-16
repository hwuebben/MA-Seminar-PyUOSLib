# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:57:40 2015

@author: JSCHOENK
"""
from __future__ import division
import numpy as np
from MPyUOSLib import BasicProcessingModule

"""
This module groups different data set handlers
"""

class Kiel(BasicProcessingModule):
    """
    Kiel data set handler.
    """
    def __init__(self, foot):
        default = {}
        BasicProcessingModule.__init__(self, foot, default)
        self.simulationSteps = 70080
        self.data = np.fromfile('DataSets/Kiel.dat', sep=' ').reshape([70080,3])
        self.output = self.data[0,:]
        
        
    def __call__(self, index=0):
        idx = np.min([index,70079])
        return self.data[idx,:]