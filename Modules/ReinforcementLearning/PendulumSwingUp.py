# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:57:40 2015

@author: JSCHOENK
"""
from __future__ import division
import numpy as np

from MPyUOSLib import BasicProcessingModule


class ValueFunction:
    def __init__(self, foot):
        self.approximator = []
        try:
            approx_input = foot["approximator"]
        except KeyError:
            approx_input = {}
        try:
            self.approxType = foot["approximator"]["kind"]
        except KeyError:
            self.approxType = "TensorExpansion"
        exec("from Modules.IncrementalLearning.LearnComponent.Approximator import " + self.approxType)
        exec("self.approximator = " + self.approxType + "(approx_input)")
        
        # setup learner
        self.learner = []
        try:
            learner_input = foot["learner"]
        except KeyError:
            learner_input = {}
        try:
            self.learnerType = foot["learner"]["name"]
        except KeyError:
            self.learnerType = "RLSTD"
        exec("from Modules.IncrementalLearning.LearnComponent.Learner import " + self.learnerType)
        exec("self.learner = " + self.learnerType + "(self.approximator,learner_input)")
        
    def evaluate(self, x):
        # Evaluation of approximator
        #
        return self.approximator(x)
        
    def evaluateD(self, x):
        # Evaluation of approximator
        #
        return self.approximator.evalD(x)
        
    def learn(self, x_0, x_1, r_1):
        phiX_0 = self.approximator.aggreg(x_0)
        phiX_1 = self.approximator.aggreg(x_1)
        self.approximator.alpha += self.learner.learn(phiX_0, phiX_1, r_1)
        

class Reward(BasicProcessingModule):
    def __init__(self, foot):
        BasicProcessingModule.__init__(self, foot)
        if not hasattr(self,"c_x"):
            self.c_x = 1.0
        if not hasattr(self,"c_u"):
            self.c_u = 0.1
        if not hasattr(self,"Tmax"):
            self.Tmax = 5.0
        
        
    def __call__(self, state, ctrl, index=0):
        out = self.c_x*(np.cos(state[0])-1)+self.c_u*2*np.log(np.cos(np.pi*(ctrl/self.Tmax)/2))/np.pi
        return out
        
class Control(BasicProcessingModule):
    def __init__(self, foot):
        BasicProcessingModule.__init__(self, foot)
        if not hasattr(self,"Tmax"):
            self.Tmax = 5.0
        if not hasattr(self,"c"):
            self.c = 1.0
        if not hasattr(self,"tau"):
            self.tau = 1.0
            
        self.ValueApprox = ValueFunction(foot)
            
    def prepare(self, antecessor):
        self.last_state = np.array(antecessor["state"].output)
        
    def evalU(self, x):
        grad = self.ValueApprox.evaluateD(x)
        m = 1.0
        l = 1.0
        out = self.Tmax*np.tanh(np.pi/2*self.tau/self.c*grad[1]/(m*l**2)+np.random.normal(scale=0.01))
        return out
        
    def __call__(self, state, reward, index=0):
        out = self.evalU(self.last_state)
        self.ValueApprox.learn(self.last_state,state,reward)
        self.last_state = np.array(state)
        return out
        
class EvalValueFnc(BasicProcessingModule):
    def __init__(self, foot):
        BasicProcessingModule.__init__(self, foot)
        
    def prepare(self, antecessor):
        self.valueFnc = antecessor["value"]
        
    def __call__(self, state, value, index=0):
        return self.valueFnc.approximator(state)
        
        
class Control2(BasicProcessingModule):
    def __init__(self, foot):
        BasicProcessingModule.__init__(self, foot)
        if not hasattr(self,"Tmax"):
            self.Tmax = 5.0
        if not hasattr(self,"c"):
            self.c = 1.0
        if not hasattr(self,"tau"):
            self.tau = 1.0
            
    def prepare(self, antecessor):
        self.LearnSys = antecessor["value"]
                
    def __call__(self, state, value, index=0):
        grad = self.LearnSys.approximator.evalD(state)
        m = 1.0
        l = 1.0
        out = self.Tmax*np.tanh(np.pi/2*self.tau/self.c*grad[1]/(m*l**2)+np.random.normal(scale=0.01))
        return out