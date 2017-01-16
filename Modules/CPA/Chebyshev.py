# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 17:40:42 2015

@author: joschnei
"""

from __future__ import division
import numpy
import matplotlib.pyplot as plt

class Chebyshev:

    def __init__(self, grade, resolution=1000, epsilon=0.1):
        self.epsilon = epsilon
        self.grade = grade
        self.res = resolution
        self.points = self.__getChebyshevPoints()
        self.points = self.points[::-1]
        self.actArray = self.__getActArray()
        self.int = self.__getIntegral()
        self.chebyDist = numpy.diff(self.points)
        self.chebyDist = numpy.insert(self.chebyDist,0,1-self.points[-1])
        self.chebyDist = numpy.append(self.chebyDist,1-self.points[-1])
        self.chebyDist = numpy.abs(self.chebyDist)
        self.chebyDistHalf = numpy.divide(self.chebyDist, 2.0)
    
    def __getCValue(self, index):
        return numpy.cos((numpy.multiply(index,2)+1)/(2*(self.grade+1))*numpy.pi)

    def __getChebyshevPoints(self):
        return self.__getCValue(range(self.grade+1))

    def __getActArray(self):
        result = []
        ar = numpy.linspace(-1,1,self.res)        
        for i in ar:
            dist = numpy.abs(numpy.subtract(self.points,i))
            minDist = numpy.min(dist)
            result.append(1/(minDist+self.epsilon))
        result = numpy.divide(result, numpy.max(result))
        return result

    def __getIntegral(self):
        return (numpy.sum(self.actArray)/self.res)

    def getLam(self, x):
        dist = numpy.abs(numpy.subtract(numpy.linspace(-1,1,self.res),x))
        idx = numpy.argmin(dist)
        return self.actArray[idx]
     
if __name__=="__main__":
    c = Chebyshev(20)
    print c.points