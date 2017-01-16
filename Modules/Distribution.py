# -*- coding: utf-8 -*-
"""
Created on Fri Mar 04 12:51:58 2016

@author: JSCHOENK
"""

from __future__ import division
import numpy as np

from MPyUOSLib import BasicProcessingModule

class Uniform(BasicProcessingModule):
    def __init__(self, foot):
        default = {"domain":[0,1], "size":1}
        BasicProcessingModule.__init__(self, foot, default)
        self.output = np.zeros(self.size)                
        
    def __call__(self,size = None, domain = None,index = 0):
        if domain is None:
            domain = self.domain
        if size is None:
            size = self.size
        return np.random.uniform(low=domain[0],high=domain[1],size=size)
        
        
        
class Gaussian(BasicProcessingModule):
    def __init__(self, foot):
        default = {"mean":0.0, "std":1.0, "size":1}
        BasicProcessingModule.__init__(self, foot, default)
        self.output = np.zeros(self.size)
        
    def __call__(self, mean = None, std = None, size = None, index = 0):
        if mean is None:
            mean = self.mean
        if std is None:
            std = self.std
        if size is None:
            size = self.size
        return np.random.normal(mean,std,size=size)



class Beta(BasicProcessingModule):
    def __init__(self,foot):
        default = {"a":2.0, "b":2.0, "domain":[0,1], "size":1}
        BasicProcessingModule.__init__(self, foot, default)
        self.output = np.zeros(self.size)
        
    def __call__(self, a = None, b = None, domain = None, size = None, index = 0):
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        if domain is None:
            domain = self.domain
        if size is None:
            size = self.size
        return np.random.beta(a,b,size=size)*(domain[1]-domain[0])+domain[0]