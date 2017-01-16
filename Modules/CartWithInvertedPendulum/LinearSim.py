# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:48:56 2016

@author: joschnei
"""

from BasicProcessingModule import BasicProcessingModule


class LinearSim(BasicProcessingModule):
    def __init__(self, foot):
        BasicProcessingModule.__init__(self, foot)
        self.foot = foot
        self.name = foot["ID"]
        self.cnt = 0
        try:
            self.maxForce = foot["maxForce"]
        except(KeyError):
            self.maxForce = 120             # 120 N is the standard F_max in "Benchmark problems for nonlinear system identification and control using Soft Computing methods: Need and overview
        self.output = 0
        

    def __call__(self, argIn, index=0):
        force = argIn*self.maxForce
        return force
    
    def end(self):
        pass
