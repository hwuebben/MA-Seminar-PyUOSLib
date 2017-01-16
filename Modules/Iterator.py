# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:57:40 2015

@author: JSCHOENK
"""
from __future__ import division
import numpy as np
import glob


from MPyUOSLib import BasicProcessingModule


"""
This module groups different iterator functions
"""

class MultiArray(BasicProcessingModule):
    """
    List iterator
    """
    def __init__(self, foot):
        default = {"array":[0]}
        BasicProcessingModule.__init__(self, foot, default)
        self.dim = len(self.array)
        self.singleArray = []
        self.lengths = np.zeros(self.dim)
        
        for i in range(self.dim):
            if type(self.array[i]) is list:
                self.singleArray.append(np.array(self.array[i]))
            else:
                exec("self.singleArray.append(" + self.array[i] + ")")
            self.lengths[i] = np.size(self.singleArray[i])
        tmpCumProd = self.lengths[::-1].cumprod()[::-1]
        self.simulationSteps = int(tmpCumProd[0])
        self.multipliers = np.ones(self.dim)
        self.multipliers[:-1] = tmpCumProd[1:]
        self.output = np.zeros(self.dim)
        
    def __call__(self, index=0):
        out = np.zeros(self.dim)
        tmpIndex = index
        for i in xrange(self.dim):
            idx = int(tmpIndex//self.multipliers[i])
            tmpIndex -= idx*self.multipliers[i]
            out[i] = self.singleArray[i][idx]
        return out

class List(BasicProcessingModule):
    """
    List iterator
    """
    def __init__(self, foot):
        default = {"list":[0]}
        BasicProcessingModule.__init__(self, foot, default)
        self.list = np.array(self.list)
    
    def __call__(self, index=0):
        index = min(index,len(self.list)-1)
        return self.list[index]

class ListMod(BasicProcessingModule):
    """
    List modulo iterator
    """
    def __init__(self, foot):
        default = {"list":[0]}
        BasicProcessingModule.__init__(self, foot, default)
        self.list = np.array(self.list)
    
    def __call__(self, index=0):
        index = np.mod(index,len(self.list))
        return self.list[index]
        
class Linspace(BasicProcessingModule):
    """
    Linspace iterator
    """
    def __init__(self, foot):
        default = {"args":[0,1,11]}
        BasicProcessingModule.__init__(self, foot, default)
        self.list = np.linspace(*self.args)
    
    def __call__(self, index=0):
        index = min(index,len(self.list)-1)
        return self.list[index]
        
class LinspaceMod(BasicProcessingModule):
    """
    Linspace modulo iterator
    """
    def __init__(self, foot):
        default = {"args":[0,1,11]}
        BasicProcessingModule.__init__(self, foot, default)
        self.list = np.linspace(*self.args)
    
    def __call__(self, index=0):
        index = np.mod(index,len(self.list))
        return self.list[index]
        
class ModuleFilesInFolder(BasicProcessingModule):
    """
    Iterates all names of ,json files in a given subdirectory of 
    ModuleDescriptions
    """
    def __init__(self, foot):
        default = {"folder":"Tutorial"}
        BasicProcessingModule.__init__(self, foot, default)
        path = 'ModuleDescription/'+ self.folder + '/*.json'
        self.iterList = glob.glob(path)
        self.nrD = len(self.iterList)
        self.simulationSteps = len(self.iterList)
        
    def prepare(self, antecessor):
        self.output = self.iterList[0]
        
    def __call__(self, index=0):
        index = min(index, len(self.iterList)-1)
        return self.iterList[index]
        