# -*- coding: utf-8 -*-
"""
Created on Sat Jan 07 18:28:40 2017

@author: Henning
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

class Sigmoid:
    def __init__(self):
        self.a = None
        self.b = np.random.normal()
    def evaluate(self,x):
        if self.a == None:
            self.a = np.random.normal(size=x.size)
        sigX = 1/( 1+np.exp( -(np.dot(self.a,x)+self.b) ) )
        
        return np.float64(sigX)
    def __str__(self):
        return "a: "+str(self.a)+" b: "+str(self.b)
        
class Gaussian:
    def __init__(self):
        self.a = None
        self.b = np.random.normal()
    def evaluate(self,x):
        if self.a == None:
            self.a = np.random.normal(size=x.size)
        d = x-self.a
        return np.exp(-self.b * np.sqrt(np.sum(np.square(d))))
class TanH:
    def __init__(self):
        self.a = np.random.normal()
        self.b = np.random.normal()
    def evaluate(self,x):
        return np.tanh()
        
if __name__=="__main__":
    af = Sigmoid()
    nrEval = 100
    xvals = np.linspace(-50,50,nrEval)
    yvals = np.empty(nrEval)
    for i,x in enumerate(xvals):
        yvals[i] = af.evaluate(x)
    plt.plot(xvals,yvals)
    plt.show()
    
