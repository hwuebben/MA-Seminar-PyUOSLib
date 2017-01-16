# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:30:30 2016

@author: joschnei
"""

from __future__ import division

import pylab as plt
import numpy as np

class PendulumPlot:
    def __init__(self, foot):
        self.foot = foot
        self.name = foot["name"]
        plt.ion()
        self.fig = plt.figure()
        self.canvas = self.fig.add_subplot(111)
        self.canvas.set_xlim([-30,30])
        self.canvas.set_ylim([-1.5,1.5])
        self.pendulumLine, = self.canvas.plot([0,0],[0,-1])
        self.textPendulumPosLabel = plt.figtext(0.92,0.7,'Theta')
        self.textPendulumPosValue = plt.figtext(0.92,0.65,'Init')   
        self.textPendulumVelLabel = plt.figtext(0.92,0.6,'dt Theta')
        self.textPendulumVelValue = plt.figtext(0.92,0.55,'Init')
        self.textCartPosLabel = plt.figtext(0.92,0.5,'Cart Position')
        self.textCartPosValue = plt.figtext(0.92,0.45,'Init')
        self.textCartVelLabel = plt.figtext(0.92,0.4,'Cart Speed')
        self.textCartVelValue = plt.figtext(0.92,0.35,'Init') 
        
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
        pendulumPos = argIn[0]
        pendulumVel = argIn[1]
        cartPos = argIn[2]
        cartVel = argIn[3]
        self.pendulumLine.set_xdata([cartPos,(np.sin(pendulumPos)+cartPos)])
        self.pendulumLine.set_ydata([0,np.cos(pendulumPos)])
        pendulumPos = np.fmod(pendulumPos, np.pi)
        self.textPendulumPosValue.set_text(str(np.around(pendulumPos,2)))
        self.textPendulumVelValue.set_text(str(np.around(pendulumVel,2)))
        self.textCartVelValue.set_text(str(np.around(cartVel,2)))
        self.textCartPosValue.set_text(str(np.around(cartPos,2)))
        plt.draw()
        plt.pause(0.0001)
        return 0
        
    def end(self):
        plt.close()
