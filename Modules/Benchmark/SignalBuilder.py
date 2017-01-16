# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:57:40 2015

@author: JSCHOENK
"""
from __future__ import division
import numpy as np
try:
    from MPyUOSLib import BasicProcessingModule
except ImportError:
    class BasicProcessingModule:
        def __init__(self,foot):            
            pass

class SignalBuilder(BasicProcessingModule):
    """
    affine linear function.
    """
    def __init__(self,foot):
        BasicProcessingModule.__init__(self,foot)
        self.output = np.zeros(1)
        
        self.segments = foot["segments"]
        self.duration = np.zeros(len(self.segments))
        for i in range(len(self.segments))        :
            self.duration[i] = self.segments[i]["duration"]
        self.cumDur = np.cumsum(self.duration)
        self.segmentOffsets = np.hstack([0,np.cumsum(self.duration)[:-1]])
        self.nrD = int(self.duration.sum())
        self.simulationSteps = self.nrD
        
    def __call__(self, index=0):
        activeSegmentIndex = np.nonzero(self.segmentOffsets<=index)[0][-1]
        segmentInternalIndex = min(index-self.segmentOffsets[activeSegmentIndex],self.duration[activeSegmentIndex])
        normalizedInternalCoord = segmentInternalIndex/(self.duration[activeSegmentIndex]-1)
        exec("out = self." + self.segments[activeSegmentIndex]["kind"] + "(activeSegmentIndex,normalizedInternalCoord)")
        return out
        
    def linear(self, index, coord):
        a = self.segments[index]["start"]
        b = self.segments[index]["final"]
        return a+(b-a)*coord
    
    def constant(self, index, coord):
        return self.segments[index]["value"]
        
    def halfcosine(self, index, coord):
        a = self.segments[index]["start"]
        b = self.segments[index]["final"]
        c = 0.5*(1-np.cos(np.pi*coord))
        return a+c*(b-a)
        
    def power(self, index, coord):
        a = self.segments[index]["start"]
        b = self.segments[index]["final"]
        power = self.segments[index]["power"]
        c = coord**power
        return c*a+(1-c)*b
        
        

if __name__ == '__main__':
    import pylab as plt
    foot = {"segments":[
                {"duration":10,"kind":"constant","value":1},
                {"duration":30,"kind":"linear","start":1,"final":4},
                {"duration":10,"kind":"halfcosine","start":4,"final":3},
                {"duration":20,"kind":"halfcosine","start":3,"final":5},
                {"duration":20,"kind":"constant","value":2}
                ]
           }
    a = SignalBuilder(foot)
    a({},0)
    x = range(100)
    y = np.zeros(np.shape(x))
    for i in range(len(x)):
        y[i] = a({},i)
        
    plt.plot(x,y)