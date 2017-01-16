# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:30:30 2016

@author: joschnei
"""

from __future__ import division

import pylab as plt
import numpy as np

class KeyAndPlot:
    def __init__(self, foot):
        self.foot = foot
        self.name = foot["name"]
        plt.ion()
        self.fig = plt.figure()
        self.canvas = self.fig.add_subplot(111)
        self.canvas.set_xlim([-30,30])
        self.canvas.set_ylim([-1.5,1.5])
        self.pendulumLine, = self.canvas.plot([0,0],[0,-1])
        self.cidPressed = self.fig.canvas.mpl_connect('key_press_event', self.keyPressed)
        self.cidReleased = self.fig.canvas.mpl_connect('key_release_event', self.keyReleased)
        self.right = 0
        self.left = 0
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
        pendulumPos = argIn[0]
        cartPos = argIn[2]
        self.pendulumLine.set_xdata([cartPos,(np.sin(pendulumPos)+cartPos)])
        self.pendulumLine.set_ydata([0,np.cos(pendulumPos)])
        plt.draw()
        plt.pause(0.0001)
        out = self.right-self.left
        return out
    
    def keyPressed(self, event):
        if event.key == 'right':
            self.right = 1
        if event.key == 'left':
            self.left = 1
        return
    
    def keyReleased(self, event):
        if event.key == 'right':
            self.right = 0
        if event.key == 'left':
            self.left = 0
        return
        
    def end(self):
        self.fig.canvas.mpl_disconnect(self.cidPressed)
        self.fig.canvas.mpl_disconnect(self.cidReleased)
        plt.close()
        pass

if __name__ == "__main__":
    kap = KeyAndPlot({"name":"Balance","state":[3.14159,0,0,0.1],"dt":0.01})
    kap([1,0,7],0)
    