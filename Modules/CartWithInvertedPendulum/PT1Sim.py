# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:48:56 2016

@author: joschnei
"""

import numpy as np

class PT1Sim:
    def __init__(self, foot):
        self.foot = foot
        self.name = foot["name"]
        self.force = 0
        self.cnt = 0
        try:
            self.maxForce = foot["maxForce"]
        except(KeyError):
            self.maxForce = 120
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
        if(argIn > 0 and self.force < 0):
            self.force = 1
            self.cnt = 0
        elif(argIn < 0 and self.force > 0):
            self.force = -1
            self.cnt = 0
        elif(argIn == 0):
            self.force = 0
            self.cnt = 0
        else:
            self.cnt += 1
            self.force = argIn*self.maxForce*(1-(1.0/(1.0+self.cnt)))
        return self.force
    
    def end(self):
        pass