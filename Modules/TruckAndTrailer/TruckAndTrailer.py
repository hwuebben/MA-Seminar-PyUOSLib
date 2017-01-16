# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:57:40 2015

@author: JSCHOENK
"""
from __future__ import division
import numpy as np

import pylab as plt


class TruckAndTrailer:

    def __init__(self,foot):
        self.foot = foot
        self.output = []
        self.name = foot["name"]
        try:
            self.dt = foot["dt"]
        except KeyError:
            self.dt = 0.01
        try:
            self.state = np.array(foot["state"])
        except KeyError:
            self.state = np.array([100,100,0,0])
            
        try:
            self.r = foot["r"]
        except KeyError:
            self.r = 3 # [m]; distance front wheel moves per time step
        
        try:
            self.L_S = foot["L_S"]
        except KeyError:
            self.L_S = 14 # [m[; length of the trailer, from rear to pivot
            
        try: 
            self.L_C = foot["L_C"]
        except KeyError:
            self.L_C = 6 # [m]; length of the cab, from pivot to front axle
            
        try:
            self.maxSteering = foot["maxSteering"]
        except KeyError:
            self.maxSteering = np.pi/2/9*7
        
        
        out = self.state.copy()
        alpha = np.arctan2(out[1],out[0])
        out[0] = alpha
        out[1] = np.sqrt(out[1]**2 + out[0]**2)
        beta = np.mod(alpha-out[2]+np.pi,2*np.pi)-np.pi
        gamma = np.mod(out[2]-out[3]+np.pi,2*np.pi)-np.pi
        
        self.extra = np.array([alpha,beta,gamma,0.0])
        self.deltaExtra = np.zeros([4])
        
    def connectInputs(self,names):
        # connect input for external force on the cart
        self.input = names[self.foot["input"]]
        
    def run(self,index):
        # read input
        argIn = self.input.output
        # perform simulation
        result = self(argIn,index)
        # provide output
        self.output = np.squeeze(result)
        
    def __call__(self,argIn,index):
        if self.state[0]>0:
            if abs(argIn)>self.maxSteering:
                argIn = self.maxSteering * np.sign(argIn)
            x = self.state[0]
            y = self.state[1]
            theta_S = self.state[2]
            theta_C = self.state[3]
            A = self.r * np.cos(argIn)
            B = A*np.cos(theta_C-theta_S)
            state_t = np.array([x - B*np.cos(theta_S) \
                               ,y - B*np.sin(theta_S) \
                               ,theta_S - np.arcsin(A*np.sin(theta_C-theta_S)/self.L_S) \
                               , theta_C + np.arcsin(self.r*np.sin(argIn)/(self.L_S+self.L_C))])
            state_t[2] = np.mod(state_t[2]+np.pi,2*np.pi)-np.pi
            state_t[3] = np.mod(state_t[3]+np.pi,2*np.pi)-np.pi
            self.state = state_t
            
        out = self.state.copy()
        alpha = np.arctan2(out[1],out[0])
        out[0] = alpha
        out[1] = np.sqrt(out[1]**2 + out[0]**2)
        beta = np.mod(out[2]-alpha+np.pi,2*np.pi)-np.pi
        gamma = np.mod(out[2]-out[3]+np.pi,2*np.pi)-np.pi
        nextExtra = np.array([alpha,beta,gamma,argIn])
        deltaExtra = np.array([alpha-self.extra[0] \
                              ,np.mod(beta-self.extra[1]+np.pi,2*np.pi)-np.pi \
                              ,gamma-self.extra[2] \
                              ,(argIn-self.extra[3])/10])
        self.extra = nextExtra
        self.deltaExtra = deltaExtra
        return out
        
    def end(self):
        # nothing to do here
        0
        
    def reset(self):
        # someday rest state here
        0
        
if __name__=='__main__':
    sim = TruckAndTrailer({"name":"TNT","state":[100,100,0,np.pi/4]})
    state = []
    nrD = 1000
    for i in range(nrD):
        state.append(sim(np.pi/4,i))
    plt.plot(range(nrD),state)
    plt.legend(['alpha','r','theta_S','theta_C'])
    plt.show()