# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:57:40 2015

@author: JSCHOENK
"""
from __future__ import division
import numpy as np

try:
    from MPyUOSLib import BasicProcessingModule
except ImportError:
    class BasicProcessingModule:
        def __init__(self,foot):
            pass
    

class SimpleMisalignmentControl(BasicProcessingModule):
    """Localization of car on a trajectory based on pose (x,y,phi)
    
    x - Position on x-axis
    y - Position on y-axis
    phi - Orientation
    
    This module localizes a car on a trajectory given by its pose (x,y,phi) and
    calculate the distance and misalignment to the trajectory.
    """
    def __init__(self, foot):
        BasicProcessingModule.__init__(self, foot)
        
        try:
            self.P = foot["P"]/np.pi
        except KeyError:
            self.P = 1.0/np.pi
        try: 
            self.phiMax = foot["maxSteering"]
        except KeyError:
            self.phiMax = np.radians(70)
        
    def prepare(self, antecessor):
        error = antecessor["error"].output
        self.output = np.array(error[2] * self.P)
        
    def __call__(self, error, index=0):
        phi = np.array(error[2] * self.P)
        return min([1, max([-1, phi / self.phiMax])]) * self.phiMax


class PPA(BasicProcessingModule):
    """Trajectory tracking control based on the Pure Pursuit Algorithm

    This module localizes a car on a trajectory given by its pose (x,y,phi)
    at the look ahead distance and calculates the steering angle to reach the 
    goal point on the trajectory.
    """
    def __init__(self, foot):
        BasicProcessingModule.__init__(self, foot)
        if not hasattr(self,"carLength"):
            self.carLength = 1.0
        if not hasattr(self,"maxSteering"):
            self.maxSteering = np.radians(70)
        
    def __call__(self, error, index=0):
        phi = np.arctan(2 * self.carLength * error[0] / (error[0]**2 + error[1]**2))
        return min([1, max([-1, phi / self.maxSteering])]) * self.maxSteering
