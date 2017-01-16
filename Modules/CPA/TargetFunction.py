# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:57:40 2015

@author: JSCHOENK
"""
from __future__ import division
import numpy as np
try:
    from BasicProcessingModule import BasicProcessingModule
except ImportError:
    class BasicProcessingModule:
        def __init__(self,foot = {}):
            pass

"""
This module groups different target functions to label scalar and low 
dimensional input signals.
"""

class linear(BasicProcessingModule):
    """
    affine linear function.
    """
    def __init__(self,foot):
        BasicProcessingModule.__init__(self,foot)
        self.output = np.zeros(1)
        
    def __call__(self, x, offset=0.0, scale=1.0, index=0):
        affin = np.hstack([np.array([1.0]),x])
        return scale*np.inner(affin,np.linspace(0,1,np.size(affin)))+offset

class cosine(BasicProcessingModule):
    """
    Normalised cosine function.
    """
    def __init__(self,foot):
        BasicProcessingModule.__init__(self,foot)
        self.output = np.zeros(1)
        
    def __call__(self, x, freq=2*np.pi, phase=0.0, index=0):
        return np.cos(freq*np.linalg.norm(x)-phase)
        
class sine(BasicProcessingModule):
    """
    Normalised sine function.
    """
    def __init__(self,foot):
        BasicProcessingModule.__init__(self,foot)
        self.output = np.zeros(1)
        
    def __call__(self, x, freq=2*np.pi, phase=0.0, index=0):
        return np.sin(freq*np.linalg.norm(x)-phase)
              
class gauss(BasicProcessingModule):
    """
    Gaussian function.
    """
    def __init__(self,foot):
        BasicProcessingModule.__init__(self,foot)
        self.output = np.zeros(1)
        
    def __call__(self, x, width=4, phase=0.0, index=0):
        r = np.linalg.norm(x)-phase
        return np.exp(-width*r**2)
        
class mexican(BasicProcessingModule):
    """
    mexican hat function.
    """
    def __init__(self,foot):
        BasicProcessingModule.__init__(self,foot)
        self.output = np.zeros(1)
        
    def __call__(self, x, width=4, phase=0.0, index=0):
        r = width*(np.linalg.norm(x)-phase)
        return (1-r**2)*np.exp((-r**2) * 0.5)
        
class step(BasicProcessingModule):
    """
    mexican hat function.
    """
    def __init__(self,foot):
        BasicProcessingModule.__init__(self,foot)
        self.output = np.zeros(1)
        
    def __call__(self, x, threshold=0.5, index=0):
        r = np.linalg.norm(x)
        return np.array(r>threshold)
        
class sigmoid(BasicProcessingModule):
    """
    mexican hat function.
    """
    def __init__(self,foot):
        BasicProcessingModule.__init__(self,foot)
        self.output = np.zeros(1)
        
    def __call__(self, x, shape=10, shift=5, index=0):
        r = shape*np.linalg.norm(x)-shift
        return np.tanh(r)
        
        
if __name__=='__main__':
    import pylab as plt
    t1 = linear({})
    t2 = cosine({})
    t3 = sine({})
    t4 = gauss({})
    t5 = mexican({})
    t6 = step({})
    t7 = sigmoid({})
    
    x = np.linspace(0,1,101)
    y1 = np.zeros(np.shape(x))
    y2 = np.zeros(np.shape(x))
    y3 = np.zeros(np.shape(x))
    y4 = np.zeros(np.shape(x))
    y5 = np.zeros(np.shape(x))
    y6 = np.zeros(np.shape(x))
    y7 = np.zeros(np.shape(x))
    for i in range(len(x)):
        y1[i] = t1(x[i],-1,2)
        y2[i] = t2(x[i])
        y3[i] = t3(x[i])
        y4[i] = t4(x[i])
        y5[i] = t5(x[i])
        y6[i] = t6(x[i])
        y7[i] = t7(x[i])

    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.plot(x,y3)
    plt.plot(x,y4)
    plt.plot(x,y5)
    plt.plot(x,y6)
    plt.plot(x,y7)
    plt.legend(["linear","cosine","sine","gauss","mexican","step","sigmoid"])
    plt.show()