# -*- coding: utf-8 -*-
"""
Created on Sun Nov 06 19:37:39 2016

@author: jhs
"""
from __future__ import division
from MPyUOSLib import BasicProcessingModule
from Modules. \
    IncrementalLearning. \
    IncrementalLearningSystem import IncrementalLearningSystem
import numpy as np

class NearestNeighbour(IncrementalLearningSystem):
    
    def __init__(self, foot):
        default = {"windowsize":50}
        IncrementalLearningSystem.__init__(self, foot, default)
        self.index = 0
        
    def prepare(self, antecessor):
        self.nrIn = np.size(antecessor["xLearn"].output)
        self.windowX = np.zeros([self.windowsize,self.nrIn])
        self.windowY = np.zeros([self.windowsize])
        
    def evaluate(self, x):
        maxIdx = np.min([self.windowsize,self.index])
        if maxIdx<1:
            return 0.0
        else:
            dist = ((self.windowX[:maxIdx]-x)**2).sum(1)
            return self.windowY[np.argmin(dist)]
        
    def learn(self, x, y):
        idx = np.mod(self.index,self.windowsize)
        self.windowX[idx,:] = x
        self.windowY[idx] = y
        self.index += 1
        
    def reset(self):
        self.windowX = np.zeros([self.windowsize,self.nrIn])
        self.windowY = np.zeros([self.windowsize])
        self.index = 0
    
class kNearestNeighbour(IncrementalLearningSystem):
    
    def __init__(self, foot):
        default = {"windowsize":50, "k":3}
        IncrementalLearningSystem.__init__(self, foot, default)
        self.index = 0
        
    def prepare(self, antecessor):
        self.nrIn = np.size(antecessor["xLearn"].output)
        self.windowX = np.zeros([self.windowsize,self.nrIn])
        self.windowY = np.zeros([self.windowsize])
        
    def evaluate(self, x):
        maxIdx = np.min([self.windowsize,self.index])
        if maxIdx<self.k:
            return 0.0
        else:
            dist = ((self.windowX[:maxIdx]-x)**2).sum(1)
            args = np.argsort(dist)
            args = args[:self.k]
            w = 1/(1+dist[args])
            out = np.inner(self.windowY[args],w)/(w.sum())
            return out
        
    def learn(self, x, y):
        idx = np.mod(self.index,self.windowsize)
        self.windowX[idx,:] = x
        self.windowY[idx] = y
        self.index += 1
        
    def reset(self):
        self.windowX = np.zeros([self.windowsize,self.nrIn])
        self.windowY = np.zeros([self.windowsize])
        self.index = 0
        
class AddSubArithmetic(BasicProcessingModule):
    def __init__(self, foot):
        BasicProcessingModule.__init__(self, foot)
        
    def __call__(self, index=0, **argIn):
        out = 0.0
        for k,v in argIn.iteritems():
            if k is '+':
                out += v
            else:
                out -= v
        return out