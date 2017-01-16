# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:57:40 2015

@author: JSCHOENK
"""
from __future__ import division
import numpy as np


from MPyUOSLib import TriggerModule


"""
This module groups different basic functions
"""

class Periodic(TriggerModule):
    """
    Periodic
    """
    def __init__(self,foot):
        TriggerModule.__init__(self,foot)
        if not hasattr(self,"period"):
            self.period = 100
        if not hasattr(self,"offset"):
            self.offset = 0

    
    def __call__(self, index=0):
        if np.mod(index-self.offset,self.period)==0:
            for i in self.target.values():
                i()

