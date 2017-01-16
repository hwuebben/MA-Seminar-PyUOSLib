# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:30:30 2016

@author: joschnei
"""

from __future__ import division

import numpy as np
from CPA import CPA

from Modules.BasicProcessingModule import BasicProcessingModule

class ControllerTDL(BasicProcessingModule):
    def __init__(self, foot):
        BasicProcessingModule.__init__(self,foot)
        self.cpa = CPA(foot["grade"], 0, foot["xmin"], foot["xmax"])
        self.output = 0
    
    def __call__(self, mode=0, x, y=0, l=0, index=0):
        if(mode == 1):
            return self.cpa.getPoint(x)
        if(mode == 2):
            self.cpa.learn(x, y, l)
        return self.cpa.getPoint(x)