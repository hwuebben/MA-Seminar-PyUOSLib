# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:57:40 2015

@author: JSCHOENK
"""
from __future__ import division
import numpy as np

from collections import deque
from os.path import basename

from MPyUOSLib import BasicProcessingModule


"""
This module groups different basic functions
"""

class Constant(BasicProcessingModule):
    """
    Unit delay for one input
    """
    def __init__(self, foot):
        default = {"value":0.0}
        BasicProcessingModule.__init__(self, foot, default)

    
    def __call__(self, index=0):
        return self.output
        
class Gain(BasicProcessingModule):
    """
    input gain
    """
    def __init__(self, foot):
        default = {"value":1.0}
        BasicProcessingModule.__init__(self, foot, default)
    
    def __call__(self, x, value=None, index=0):
        if value is None:
            value = self.value
        return x*value

class UnitDelay(BasicProcessingModule):
    """
    Unit delay for one input
    """
    def __init__(self, foot):
        BasicProcessingModule.__init__(self, foot)
        self.delay = []
        
    def prepare(self, antecessor):
        prev_out = antecessor["value"].output
        self.delay = np.zeros(np.shape(prev_out))
        self.output = self.delay
        
    def __call__(self, value, index=0):
        self.output = self.delay
        self.delay = value
        return self.output
        
class kDelay(BasicProcessingModule):
    """
    k delay for one input
    """
    def __init__(self, foot):
        default = {"delay":1}
        BasicProcessingModule.__init__(self, foot, default)
        
    
    def prepare(self, antecessor):
        prev_out = antecessor["value"].output
        if self.delay > 0:
            self.circbuffer = deque(maxlen=self.delay)
            self.circbuffer.append(prev_out)
            self.output = self.circbuffer[0]
        else:
            self.output = prev_out
        
    def __call__(self, value, index=0):
        if self.delay > 0:
            self.output = self.circbuffer[0]
            self.circbuffer.append(value)
        else:
            self.output = value
        
        return self.output

        
class ZeroOrderHold(BasicProcessingModule):
    """
    Zero order hold
    """
    def __init__(self, foot):
        BasicProcessingModule.__init__(self, foot)
        self.output = np.zeros(1)
        try:
            self.interval = int(foot["interval"])
        except KeyError:
            self.interval = 2
        self.hold = np.zeros(1)
        
    def __call__(self, value, index=0):
        if np.mod(index,self.interval)==0:
            self.hold = np.array(value)
        return self.hold
        
        
class MultiDelayReadParallel(BasicProcessingModule):
    """
    Variable delay unit with parallel read out for scalar data streams
    """
    def __init__(self, foot):
        BasicProcessingModule.__init__(self, foot)
        self.output = np.zeros(1)
        self.delaySteps = foot["DelaySteps"]
        self.order = max(self.delaySteps)+1
        self.delay = []
        
        
    def __call__(self, value, index=0):
        self.delay.insert(0, value)
        if len(self.delay)>self.order:
            self.delay.pop([-1])
        if len(self.delay)<self.order:
            self.output = np.zeros(np.shape(self.delaySteps))
        else:
            self.output = self.delay[self.delaySteps]

        return self.output


class BusCreator(BasicProcessingModule):
    """
    Group many scalar inputs to one vector
    """
    def __init__(self, foot):
        BasicProcessingModule.__init__(self, foot)
        self.nrIn = len(self.input)
        self.output = np.zeros(self.nrIn)
        
        
    def prepare(self, antecessor):
        self.order = self.input.keys()
        self.order.sort()
        
    def __call__(self, index=0, **argIn):
        out = np.zeros(len(self.input))
        for i in range(self.nrIn):
            out[i] = argIn[self.order[i]]
        return out
        
class BusSelector(BasicProcessingModule):
    """
    Group many scalar inputs to one vector
    """
    def __init__(self, foot):
        default = {"index":[]}
        BasicProcessingModule.__init__(self, foot, default)
        
        
    def prepare(self, antecessor):
        self.output = antecessor["x"].output[self.index]
        
    def __call__(self, x, index=0):
        return x[self.index]
        
        
class Concat(BasicProcessingModule):
    """
    Group many scalar inputs to one vector
    """
    def __init__(self, foot):
        BasicProcessingModule.__init__(self, foot)
        self.output = np.zeros(1)
        
        
    def __call__(self, index=0, **argIn):
        if argIn:
            return np.hstack(argIn.values())
        else:
            return np.zeros(1)

            
class Summation(BasicProcessingModule):

    def __init__(self, foot):
        BasicProcessingModule.__init__(self, foot)
        
        
    def __call__(self, index=0, **argIn):
        return np.array(argIn.values()).sum()

        
class Product(BasicProcessingModule):

    def __init__(self, foot):
        BasicProcessingModule.__init__(self, foot)
        self.output = np.ones(1)
        
        
    def __call__(self, index=0, **argIn):
        return np.array(argIn.values()).prod()
        
class ModuleAttributeSelector(BasicProcessingModule):
    def __init__(self, foot):
        BasicProcessingModule.__init__(self, foot)
        
    def prepare(self, antecessor):
        self.ref = antecessor.values()[0]
        self.output = getattr(self.ref, antecessor.keys()[0])
        
    def __call__(self, index=0, **argIn):
        return getattr(self.ref, argIn.keys()[0])
        
class strcutFilename(BasicProcessingModule):
    def __init__(self, foot):
        BasicProcessingModule.__init__(self, foot)
        self.output = ''
        
    def prepare(self, antecessor):
        if antecessor.has_key("fullfile"):
            fullfile = antecessor["fullfile"].output
            b = basename(fullfile)        
            index_of_dot = b.index('.')
            out = b[:index_of_dot]
            self.output = out
        
    def __call__(self, fullfile, index=0):
        return self.output