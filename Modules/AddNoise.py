# -*- coding: utf-8 -*-
"""
Created on Fri Mar 04 12:51:58 2016

@author: JSCHOENK
"""

from __future__ import division
import numpy as np

from MPyUOSLib import BasicProcessingModule

"""
This module groups different probability distribution functions acting as
additive noise to signals.
"""

class Uniform(BasicProcessingModule):
    """
    Adds symetric uniformly distributed noise of given width to the signal.
    """
    def __init__(self,foot):
        default = {"width":1.0}
        BasicProcessingModule.__init__(self, foot, default)
        if hasattr(self, "range"):
            self.width = self.range
        
    def __call__(self, value, width=None, index = 0):
        if width is None:
            width = self.width
        return value + (np.random.uniform(size=np.shape(value)) - 0.5)*2.0*width
        
        
class Gaussian(BasicProcessingModule):
    """
    Adds normaly distributed noise with zero mean and given variance to the 
    signal.
    """
    def __init__(self,foot):
        default = {"std":1.0}
        BasicProcessingModule.__init__(self, foot, default)
    
    def prepare(self, antecessor):
        value = antecessor["value"].output
        self.output = np.random.normal(0.0, self.std, size=np.shape(value))
    
    def __call__(self, value, std = None, index = 0, **kw):
        if std is None:
            std = self.std
        if std > 0.0:
            return value + np.random.normal(0.0, std, size=np.shape(value))
        else:
            return value


class Beta(BasicProcessingModule):
    """
    Adds noise of given width to the signal according to a beta disribution.
    """
    def __init__(self, foot):
        default = {"width":1.0, "a":2.0, "b":2.0}
        BasicProcessingModule.__init__(self, foot, default)
        if hasattr(self, "range"):
            self.width = self.range
        
    def __call__(self, value, a=None, b=None, width=None, index = 0):
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        if width is None:
            width = self.width
        return value + (np.random.beta(a, b, size=np.shape(value)) - 0.5)*2.0*width