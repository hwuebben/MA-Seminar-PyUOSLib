# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:57:40 2015

@author: JSCHOENK
"""
from __future__ import division
import pylab as plt
import numpy as np

class plotTNT:

    def __init__(self,foot):
        self.foot = foot
        self.dataX = []
        self.dataX2 = []
        self.dataY = []
        self.dataZ = []
        self.name = foot["name"]
        plt.ion()
        
    def connectInputs(self,names):
        
        self.inputY = names[self.foot["state"]]
        
        
    def run(self,index):
        self.dataX.append(self.inputY.extra)
        self.dataX2.append(self.inputY.deltaExtra)
        
        
        
        state = self.inputY.output
#        stateX = self.inputY.state[0]
#        stateY = self.inputY.state[1]
#        alpha = state[0]
#        r = state[1]
        L_S = self.inputY.L_S
        L_C = self.inputY.L_C
#        p0 = np.array([stateX,stateY])
        p0 = np.zeros([2])
        p0[0] = self.inputY.state[0]
        p0[1] = self.inputY.state[1]
        d1 = np.array([L_S*np.cos(state[2]),L_S*np.sin(state[2])])
        p1 = p0+d1
        d2 = np.array([L_C*np.cos(state[3]),L_C*np.sin(state[3])])
        p2 = p1+d2
        x = [p0[0],p1[0],p2[0]]
        y = [p0[1],p1[1],p2[1]]
        self.dataY.append(x)
        self.dataZ.append(y)
##        plt.subplot(2, 1, 1)
##        plt.plot(x,y)

        
        
#        plt.plot(x2, y2, 'r.-')
#        plt.xlabel('time (s)')
#        plt.ylabel('Undamped')

        
        
    def end(self):
        ax = plt.subplot(3, 1, 1)
        for i in range(len(self.dataY)):
            plt.plot(self.dataY[i],self.dataZ[i])
#            plt.axes().set_aspect('equal', 'datalim')            
        ax.set_aspect('equal','datalim')
        
        plt.subplot(3, 1, 2)        
        plt.plot(range(len(self.dataX)),np.array(self.dataX).squeeze())
        plt.plot(range(len(self.dataX)),np.ones(len(self.dataX))*np.pi/180*20,'--')
        
        plt.legend(["alpha","beta","gamma"])
        plt.subplot(3, 1, 3)
        plt.plot(range(len(self.dataX2)),np.array(self.dataX2).squeeze())
        plt.show()
        plt.draw()

#        if self.inputX:
#            plt.plot(self.dataX,self.dataY,'.')
#        else:
#            print plt.shape(self.dataY)
#            plt.plot(range(len(self.dataY)),self.dataY)
#        try:
#            plt.legend(self.foot["legend"])
#        except KeyError:
#            1
#        plt.show()
    
        
    def reset(self):
        pass