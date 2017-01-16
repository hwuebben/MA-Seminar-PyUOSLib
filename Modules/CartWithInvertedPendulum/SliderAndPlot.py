# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:30:30 2016

@author: joschnei
"""

from __future__ import division

import pylab as plt
import numpy as np
from matplotlib.widgets import Slider

from BasicProcessingModule import BasicProcessingModule

class SliderAndPlot(BasicProcessingModule):
    def __init__(self, foot):
        BasicProcessingModule.__init__(self, foot)
        self.foot = foot
        self.name = foot["ID"]
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
        axcolor ='lightgoldenrodyellow'
        self.sliderAxis = plt.axes([0.25, 0.01, 0.65, 0.03], axisbg=axcolor)
        self.sliderForce = Slider(self.sliderAxis, 'Force', -1, 1, valinit=0)
        self.textPendulumPosLabel = plt.figtext(0.92,0.7,'Theta')
        self.textPendulumPosValue = plt.figtext(0.92,0.65,'Init')   
        self.textPendulumVelLabel = plt.figtext(0.92,0.6,'dt Theta')
        self.textPendulumVelValue = plt.figtext(0.92,0.55,'Init') 
        self.textCartVelLabel = plt.figtext(0.92,0.5,'Cart Speed')
        self.textCartVelValue = plt.figtext(0.92,0.45,'Init') 
        
    
    def __call__(self, argIn, index=0):
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
        plt.draw()
        plt.pause(0.0001)
        out = self.right-self.left
        out = self.sliderForce.val
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
    kap = SliderAndPlot({"name":"Balance","state":[3.14159,0,0,0.1],"dt":0.01})
    
