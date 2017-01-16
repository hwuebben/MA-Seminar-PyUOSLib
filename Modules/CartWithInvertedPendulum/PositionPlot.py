# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 11:46:29 2016

@author: joschnei
"""

from __future__ import division

import pylab as plt

class PositionPlot:
    def __init__(self, foot):
        self.foot = foot
        self.name = foot["name"]
        self.posArray = []
        self.speedArray = []
        
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
        self.posArray.append(argIn[2])
        self.speedArray.append(argIn[3])
        return
    
    def end(self):
        plt.figure()
        plt.plot(self.speedArray)
        plt.plot(self.posArray)
        plt.legend(['Speed', 'Position'])
        pass