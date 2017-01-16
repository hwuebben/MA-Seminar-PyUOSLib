# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:57:40 2015

@author: JSCHOENK
"""
from __future__ import division
import numpy as np

from MPyUOSLib import BasicProcessingModule

"""
This module groups different dynamic processess for modelling and control
"""

class BicycleModel(BasicProcessingModule):
    """Bicycle model for simulating ackermann steering
    
    """
    def __init__(self,foot):
        BasicProcessingModule.__init__(self,foot)
        
        self.x = 0
        self.y = 0
        self.phi = 1

        if not hasattr(self,"vel"):
            self.vel = 1.0
        if not hasattr(self,"carLength"):
            self.carLength = 1.0
            
        self.output = np.array([self.x,self.y,self.phi])
        
    def __call__(self, steering, velIn=None, index=0):
        if velIn:
            self.vel = velIn
        
        if steering == 0.0:
            dphi = 0
            dx_car = self.vel
            dy_car = 0
        else:
            dphi = self.vel * np.tan(steering) / self.carLength
            r = self.carLength / np.tan(steering)
            dy_car = r * (1 - np.cos(dphi))
            dx_car = r * np.sin(dphi)
        dx = dx_car * np.cos(self.phi) - dy_car * np.sin(self.phi)
        dy = dx_car * np.sin(self.phi) + dy_car * np.cos(self.phi)
        self.x += dx
        self.y += dy
        self.phi = np.mod(self.phi + dphi, 2 * np.pi)    # bearing angle must be in [0;2Pi]

        return np.array([self.x,self.y,self.phi])
