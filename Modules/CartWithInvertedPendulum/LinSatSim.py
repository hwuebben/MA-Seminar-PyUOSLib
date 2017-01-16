# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:48:56 2016

@author: joschnei
"""

import numpy as np

class LinSatSim:
    def __init__(self, foot):
        self.foot = foot
        self.name = foot["name"]
        self.force = 0
        try:
            self.maxForce = foot["maxForce"]
        except(KeyError):
            self.maxForce = 30
        self.output = 0
        
    def connectInputs(self,names):
        # connect input for external force on the cart
        self.input = names[self.foot["input"]]
        
    def run(self,index):
        # read input
        argIn = self.input.output
        # perform simulation
        result = self(argIn,index)
        # provide output
        self.output = result

    def __call__(self, argIn, index):
        self.force = np.maximum(np.minimum(self.force+argIn, self.maxForce), -self.maxForce)
        return self.force
    
    def end(self):
        pass