# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:57:40 2015

@author: JSCHOENK
"""
from __future__ import division
import numpy as np
from MPyUOSLib import BasicProcessingModule

"""
This module groups different target functions to label scalar and low 
dimensional input signals.
"""

class Linear(BasicProcessingModule):
    """
    affine linear function.
    """
    def __init__(self, foot):
        default = {"offset":0.0, "scale":1.0}
        BasicProcessingModule.__init__(self, foot, default)
        
    def __call__(self, x, offset=None, scale=None, index=0):
        if offset is None:
            offset = self.offset
        if scale is None:
            scale = self.scale
        affin = np.hstack([np.array([1.0]),x])
        return scale*np.inner(affin,np.linspace(0,1,np.size(affin)))+offset

class Cosine(BasicProcessingModule):
    """
    Norm-based cosine function.
    """
    def __init__(self, foot):
        default = {"freq":2*np.pi, "phase":0.0}
        BasicProcessingModule.__init__(self, foot, default)
        
    def __call__(self, x, freq=None, phase=None, index=0):
        if freq is None:
            freq = self.freq
        if phase is None:
            phase = self.phase
        return np.cos(freq*np.linalg.norm(x)-phase)
        
        
class Sine(BasicProcessingModule):
    """
    Norm-based sine function.
    """
    def __init__(self, foot):
        default = {"freq":2*np.pi, "phase":0.0}
        BasicProcessingModule.__init__(self, foot, default)
        
    def __call__(self, x, freq=None, phase=None, index=0):
        if freq is None:
            freq = self.freq
        if phase is None:
            phase = self.phase
        return np.sin(freq*np.linalg.norm(x)-phase)
              
class Gauss(BasicProcessingModule):
    """
    Norm-based gaussian function.
    """
    def __init__(self, foot):
        default = {"width":4.0, "phase":0.0}
        BasicProcessingModule.__init__(self, foot, default)
        
        
    def __call__(self, x, width=None, phase=None, index=0):
        if width is None:
            width = self.width
        if phase is None:
            phase = self.phase
        r = np.linalg.norm(x)-phase
        return np.exp(-(r/width)**2)
        
class Mexican(BasicProcessingModule):
    """
    Norm-based mexican hat function.
    """
    def __init__(self, foot):
        default = {"width":4.0, "phase":0.0}
        BasicProcessingModule.__init__(self, foot, default)
        
    def __call__(self, x, width=None, phase=None, index=0):
        if width is None:
            width = self.width
        if phase is None:
            phase = self.phase
        r = width*(np.linalg.norm(x)-phase)
        return (1-r**2)*np.exp((-r**2) * 0.5)
        
class Step(BasicProcessingModule):
    """
    Norm-based step function.
    """
    def __init__(self, foot):
        default = {"threshold":0.5, "height":1.0}
        BasicProcessingModule.__init__(self, foot, default)
        
    def __call__(self, x, threshold=None, height=None, index=0):
        if threshold is None:
            threshold = self.threshold
        if height is None:
            height = self.height
        r = np.linalg.norm(x)
        return height*np.array(r>threshold)
        
class Sigmoid(BasicProcessingModule):
    """
    Norm-based sigmoid function.
    """
    def __init__(self, foot):
        default = {"shape":10.0, "shift":5.0, "height":1.0}
        BasicProcessingModule.__init__(self, foot, default)
        
    def __call__(self, x, shape=None, shift=None, height=None, index=0):
        if shape is None:
            shape = self.shape
        if shift is None:
            shift = self.shift
        if height is None:
            height = self.height
        r = shape*np.linalg.norm(x)-shift
        return np.tanh(r)*height
        
class Kink(BasicProcessingModule):
    """
    Norm-based kink function.
    """
    def __init__(self, foot):
        default = {"level":0.0, "shift":0.5, "slope":1.0}
        BasicProcessingModule.__init__(self, foot, default)
        
    def __call__(self, x, level=None, shift=None, slope=None, index=0):
        if level is None:
            level = self.level
        if shift is None:
            shift = self.shift
        if slope is None:
            slope = self.slope
        r = np.linalg.norm(x)-shift
        out = level
        if r>shift:
            out += (r-shift)*slope
        return out
        
class Parabola(BasicProcessingModule):
    """
    Norm-based parabola function.
    """
    def __init__(self, foot):
        default = {"level":0.0, "shift":0.5, "scale":1.0}
        BasicProcessingModule.__init__(self, foot, default)
        
    def __call__(self, x, level=None, shift=None, scale=None, index=0):
        if level is None:
            level = self.level
        if shift is None:
            shift = self.shift
        if scale is None:
            scale = self.scale
        r = np.linalg.norm(x)-shift
        return level + scale*r**2
        
class Hyperbola(BasicProcessingModule):
    """
    Norm-based hyperbola function.
    """
    def __init__(self, foot):
        default = {"level":0.0, "shift":0.5, "scale":1.0}
        BasicProcessingModule.__init__(self, foot, default)
        
    def __call__(self, x, level=None, shift=None, scale=None, index=0):
        if level is None:
            level = self.level
        if shift is None:
            shift = self.shift
        if scale is None:
            scale = self.scale
        r = np.linalg.norm(x)-shift
        return level + scale*(1/r)
        
class Absolute(BasicProcessingModule):
    """
    Norm-based parabola function.
    """
    def __init__(self, foot):
        default = {"level":0.0, "shift":0.5, "scale":1.0}
        BasicProcessingModule.__init__(self, foot, default)
        
    def __call__(self, x, level=None, shift=None, scale=None, index=0):
        if level is None:
            level = self.level
        if shift is None:
            shift = self.shift
        if scale is None:
            scale = self.scale
        r = np.linalg.norm(x)-shift
        return level + scale*abs(r)
        
class Tan(BasicProcessingModule):
    """
    Norm-based tangent function.
    """
    def __init__(self, foot):
        default = {"level":0.0, "shift":0.5, "scale":1.0}
        BasicProcessingModule.__init__(self, foot, default)
        
    def __call__(self, x, level=None, shift=None, scale=None, index=0):
        if level is None:
            level = self.level
        if shift is None:
            shift = self.shift
        if scale is None:
            scale = self.scale
        r = np.linalg.norm(x)-shift
        return level + scale*np.tan(r)
        
class Power(BasicProcessingModule):
    """
    Norm-based power function.
    """
    def __init__(self, foot):
        default = {"level":0.0, "shift":0.5, "scale":1.0, "power": 1.0}
        BasicProcessingModule.__init__(self, foot, default)
        
    def __call__(self, x, level=None, shift=None, scale=None, power=None, index=0):
        if level is None:
            level = self.level
        if shift is None:
            shift = self.shift
        if scale is None:
            scale = self.scale
        if power is None:
            power = self.power
        r = np.linalg.norm(x)-shift
        return level + scale*r**power
        
        
class Dim1TypeA(BasicProcessingModule):
    """
    One dimensional target function.
    """
    def __init__(self, foot):
        BasicProcessingModule.__init__(self, foot)
        
    def __call__(self, x, index=0, **kw):
        return np.cos(x**2)/(0.1*x**2)