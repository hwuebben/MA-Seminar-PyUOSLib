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
        self.a = np.random.normal()
        self.b = np.random.normal()
    def evaluate(self,x):
        sigX = 1/( 1+np.exp( -(self.a * x+self.b) ) )
        
        return np.float64(sigX)
    def __str__(self):
        return "a: "+str(self.a)+" b: "+str(self.b)
        
class Gaussian:
    def __init__(self):
        self.a = np.random.normal()
        self.b = np.random.normal()
    def evaluate(self,x):
        return np.exp(-self.b * np.abs(x-self.a))
        
class TanH:
    
    def evaluate(self,x):
        return np.tanh(x)
class QuaRBF:
    def __init__(self):
        self.a = np.random.normal()
        self.b = np.random.normal()
    
    def evaluate(self,x):
        return np.sqrt(np.abs(x-self.a) + self.b**2)
class CosFourier:
    def __init__(self):
        self.a = np.random.normal()
        self.b = np.random.normal()
    def evaluate(self,x):
        return np.cos(self.a*x+self.b)
class HyperTan:
    def __init__(self):
        self.a = np.random.normal()
        self.b = np.random.normal()
    def evaluate(self,x):
        out = (1-np.exp(-(self.a*x+self.b))) / (1+np.exp(-(self.a*x+self.b)))
        return out
def invSVD(toInv,rcond):
    
        U, s, Vh = np.linalg.svd(toInv, full_matrices=False)
        singular = s < rcond
        if singular.any():
            print("singular value detected: ",s)
        nans = np.isnan(s)
        invS = 1/s
        invS[singular] = 0
        invS[nans] = 0
        inverse = np.dot(np.dot(U,np.diag(invS)),Vh)
        return inverse
        


if __name__=="__main__":
    af = HyperTan()
    nrEval = 100
    xvals = np.linspace(-50,50,nrEval)
    yvals = np.empty(nrEval)
    for i,x in enumerate(xvals):
        yvals[i] = af.evaluate(x)
    plt.plot(xvals,yvals)
    plt.show()
    
