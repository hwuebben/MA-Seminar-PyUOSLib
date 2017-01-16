# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 21:31:27 2017

@author: Henning
"""
from __future__ import division
from Modules.IncrementalLearning.IncrementalLearningSystem import IncrementalLearningSystem 
from FOS_ELM import FOS_ELM
import numpy as np

class FOS_ENS_ELM ( IncrementalLearningSystem ):
    
    def init ( self , foot ): 
        self.foot= foot
        IncrementalLearningSystem.init( self , foot )

    def prepare ( self , antecessor ): 
        nrHidden = 10
        windowSize = 100
        self.ensemble_size = 3

        self.nrIn = len(antecessor["xLearn"].output)# + 1
        self.nrInit = int(nrHidden *1.5)
        self.ensemble = np.empty(self.ensemble_size,dtype=object)
        
        for i in range(self.ensemble_size):
            self.ensemble[i] = FOS_ELM(self.nrIn,nrHidden,windowSize,self.nrInit)
        self.xdomain = np.ones((2,self.nrIn))
        self.xdomain[0] = self.xdomain[0] * -np.inf
        self.xdomain[1] = self.xdomain[1] * np.inf
        self.startedLearning = False
        self.counter = 0
        #print(self.ensemble)
            
    def normalize(self,x):
        x = x/10
        #x = (x-3.12) * (5/3)
        
#        xn = np.ones(self.nrIn)
#        xn[1::] = x
        return x
        
    def evaluate(self,x):
        if not self.startedLearning:
            return np.array([[0]])
        x=self.normalize(x)
        #print(x)
        out = 0
        outs = []
        for i in range(self.ensemble_size):
            outi = self.ensemble[i].evaluate(x)
            out += outi / self.ensemble_size
            outs.append(outi)
            #outs.append( self.ensemble[i].evaluate(x) )
            #print(self.ensemble[i].evaluate(x))
        #print(outs)
        
        return out
    
    def learn(self,x,y):
        #print(x)
        for i,xi in enumerate(x):
            if self.xdomain[0][i] < xi:
                self.xdomain[0][i] = xi
                print(self.xdomain)
            if self.xdomain[1][i] > xi:
                self.xdomain[1][i] = xi
                print(self.xdomain)
        x=self.normalize(x)
        #print(x)
        for i in range(self.ensemble_size):
            self.ensemble[i].learn(x,y)
        self.counter += 1
        if self.counter == self.nrInit:
            self.startedLearning = True
            
    def reset(self):
        for ens in self.ensemble:
            ens.reset()
        self.ensemble = np.empty(self.ensemble_size)
        
        