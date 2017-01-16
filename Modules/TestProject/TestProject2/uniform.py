# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:57:40 2015

@author: JSCHOENK
"""
from __future__ import division
import numpy as np

from Modules.BasicProcessingModule import BasicProcessingModule

class uniform(BasicProcessingModule):
    """
    Data generator yielding random numbers drawn according to a uniform 
    distribution
    """
    def __init__(self,foot):
        BasicProcessingModule.__init__(self,foot)
        self.output = np.random.rand()
        
    def __call__(self,argIn = 0,index = 0):
        return np.random.rand()
        
if __name__=="__main__":
    foot = {"name":"Test1"}
    u = uniform(foot)
    print u()