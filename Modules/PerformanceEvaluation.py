# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:57:40 2015

@author: JSCHOENK
"""
from __future__ import division
import numpy as np

from MPyUOSLib import BasicProcessingModule

"""
This module groups different delay functions
"""

class CumulativeLoss(BasicProcessingModule):
    """
    This module provides basic cumulaive loss performance evaluation

    For scalar values:
    - squared loss
    - absolute loss
    
    For vector values:
    - norm of given order
    """
    def __init__(self,foot):
        default = {"measure":"square", "order":2}
        BasicProcessingModule.__init__(self, foot, default)
        self.measure = str(self.measure)
        if self.measure == "square":
            self.__call__ = self.square
        elif self.measure == "absolute":
            self.__call__ = self.absolute    
        elif self.measure == "norm":
            self.__call__ = self.norm
        else:
            print "error: unknown measure:", self.measure
            
        self.reset()
        
    def reset(self):
        self.CL = 0.0
        self.output = np.zeros(1)
        
    def square(self, target, prediction, index = 0):
        error = (prediction-target)**2
        self.CL += error
        return np.array(self.CL)
        
    def absolute(self, target, prediction, index = 0):
        error = abs(prediction-target)
        self.CL += error
        return np.array(self.CL)
    
    def norm(self, target, prediction, index = 0):
        error = np.linalg.norm(prediction-target, ord=self.order)
        self.CL += error
        return np.array(self.CL)

class GroundTruthLoss(BasicProcessingModule):
    """
    This module provides mean squared error 
    ground truth loss preformance evaluation
    """
    def __init__(self,foot):
        BasicProcessingModule.__init__(self,foot)
        self.reset()
        
    def reset(self):
        self.output = np.zeros(1)
        
    def prepare(self,antecessor):
        print "Warning TODO: Implement GroundTruthLoss, yields zero"
        
        
    def __call__(self, target, prediction, index = 0):        
        return np.zeros(1)