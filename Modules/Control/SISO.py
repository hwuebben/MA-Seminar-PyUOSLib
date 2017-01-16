# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:57:40 2015

@author: JSCHOENK
"""
from __future__ import division
import numpy as np

from MPyUOSLib import BasicProcessingModule


class FeedbackLinSin(BasicProcessingModule):
    def __init__(self, foot):
        BasicProcessingModule.__init__(self, foot)
        if not hasattr(self,"a_m"):
            self.a_m = 1.0
        if not hasattr(self,"target"):
            self.target = 0.0
        
        
    def __call__(self, state, target=None, index=0):
        if target is None:
            target = self.target
        out = -self.a_m*(state-target)-np.sin(state)
        return out