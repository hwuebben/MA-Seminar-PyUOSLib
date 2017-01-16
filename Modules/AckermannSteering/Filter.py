# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 13:06:49 2016

@author: MGRESHAKE
"""
from __future__ import division
from collections import deque

import numpy as np

from MPyUOSLib import BasicProcessingModule

class AddGaussianNoisePose(BasicProcessingModule):
    """
    Adds normaly distributed noise with zero mean and given variance to the 
    signal.
    """
    def __init__(self,foot):
        BasicProcessingModule.__init__(self,foot)
        if not hasattr(self,"std"):
            self.std = 1.0
    
    def prepare(self, antecessor):
        value = antecessor["value"].output
        self.output = np.random.normal(0.0,self.std,size=np.shape(value))
    
    def __call__(self, value, std = [], index = 0, **kw):
        if not std:
            std = self.std
        out = value + np.random.normal(0.0,std,size=np.shape(value))
        out[2] = np.mod(out[2],2*np.pi)
        return out


class Kalman(BasicProcessingModule):
    """ Kalman filter for state estimation of Ackermann-steered vehicles
    
    x - estimate state
    P - estimate error covariance
    F - dynamic model
    H - observation model
    Q - process noise covariance
    R - measurement noise covariance
    
    This module estimates the state of an Ackermann-steered vehicle under 
    delayed and noisy measurements with a special implementation of the Kalman
    filter.
    """
    def __init__(self, foot):
        BasicProcessingModule.__init__(self, foot)
        
        if not hasattr(self,"vel"):
            self.vel = 1.0
        if not hasattr(self,"carLength"):
            self.carLength = 1.0
        try:
            self.k = foot["delay"]
        except KeyError:
            pass

        self.x = np.array([0,0,1])
        self.P = np.eye(3)
        self.F = np.eye(3)
        self.H = np.eye(3)
        self.Q = 0.0001 * self.v * np.eye(3)
        self.R = 10 * np.eye(3)
        
        self.delay = 0
        self.ctrl = deque(maxlen=self.k)
        self.output = self.x

    def __call__(self, state, steering, index=0):
        # Prediction AND correction
        if self.delay == self.k:
            # Update model
            B = np.array([[np.cos(self.x[2]), 0],
                          [np.sin(self.x[2]), 0],
                          [0, 1 / self.carLength]])                          # control model
            u = np.array([self.vel, self.vel * np.tan(self.ctrl[0])])    # control variable
            
            # Project state ahead
            self.x = np.dot(self.F, self.x) + np.dot(B, u)

            # Project error covariance ahead
            self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
            
            # Compute kalman gain
            S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R      # innovation covariance
            K = np.dot(np.dot(self.P, self.H.T), np.linalg.pinv(S))    # kalman gain           
            
            # Update state
            y = state - np.dot(self.H, self.x)
            y[2] = np.mod(y[2] + np.pi, 2 * np.pi) - np.pi
            self.x = self.x + np.dot(K, y)

            # Update error covariance
            self.P = np.dot((np.eye(3) - np.dot(K, self.H)), self.P)
        else:
            self.delay += 1
        
        self.ctrl.append(steering)
        self.x[2] = np.mod(self.x[2], 2 * np.pi)    # bearing angle must be in [0,2Pi] 
        self.output = self.x
        
        # Prediction ONLY
        for i in range(self.delay):
            # Update model
            B = np.array([[np.cos(self.output[2]), 0],
                          [np.sin(self.output[2]), 0],
                          [0, 1 / self.carLength]])
            u = np.array([self.vel, self.vel * np.tan(self.ctrl[i])])
            
            # Project state ahead
            self.output = np.dot(self.F, self.output) + np.dot(B, u)
            self.output[2] = np.mod(self.output[2], 2 * np.pi)    # bearing angle must be in [0,2Pi] 
        
        return self.output


class Prefix(BasicProcessingModule):
    """ Adaptive Model-based State Prediction
    
    This module estimates state and parameters with a finite impulse response 
    filter and an one step ahead prediciton for model adaptation based on 
    delayed inputs.    
    """
    def __init__(self, foot):
        BasicProcessingModule.__init__(self, foot)
        
        self.foot = foot
#        self.predictionHorizon = foot["predictionHorizon"]
#        self.nrState = foot["nrState"]
        if not hasattr(self,"vel"):
            self.vel = 0.3
        if not hasattr(self,"carLength"):
            self.carLength = 1.0
        if not hasattr(self,"delay"):
            pass
        if not hasattr(self,"delayFIR"):
            self.delayFIR = 3
        
        self.total_delay = 0
        self.fltr1 = deque(maxlen=2*self.delayFIR+1)
        self.fltr2 = deque(maxlen=2*self.delayFIR+1)
        self.ctrl = deque(maxlen=self.delay+self.delayFIR)
        self.output = np.array([0,0,1])
        
    def prepare(self,antecessor):
        self.measure = (antecessor["x"].output)
        self.signal = antecessor["u"].output
        self.nrState = len(self.measure)
        
        # setup approximators
        self.approximator = []
        try:
            approx_input = self.foot["approximator"]
        except KeyError:
            approx_input = {}
        try:
            self.approxType = self.foot["approximator"]["kind"]
        except KeyError:
            self.approxType = "TensorExpansion"
        exec("from Modules.IncrementalLearning.LearnComponent.Approximator import " + self.approxType)
        for i in range(self.nrState):
            exec("self.approximator.append(" + self.approxType + "(approx_input))")

        # setup learners
        self.learner = []
        try:
            learner_input = self.foot["learner"]
        except KeyError:
            learner_input = {}
        try:
            self.learnerType = self.foot["learner"]["name"]
        except KeyError:
            self.learnerType = "LRLS"
        exec("from Modules.IncrementalLearning.LearnComponent.Learner import " + self.learnerType)
        for i in range(self.nrState):
            exec("self.learner.append(" + self.learnerType + "(self.approximator[i],learner_input))")

    def fir(self, y):
        # finite impulse response filter with polynomial interpolation
        if len(y) > 1:
            n = self.width - 1
            x = np.linspace(-1, 1, len(y))
            z = np.polyfit(x, y, n)
        else:
            z = y

        return np.polyval(z, 0)
       
    def evaluate(self, x, u):
        # Evaluation of approximators
        x_in = np.hstack([x,u])
        out = np.zeros(self.nrState)
        for i in range(self.nrState):
            out[i] = self.approximator[i](x_in)
        return out

    def learn(self, x, y):
        phiX = self.approximator[0].aggreg(x)
        for i in range(self.nrState):
            yp = self.approximator[i].evaluatePhiX(phiX)
            self.approximator[i].alpha += self.learner[i].learn(x, phiX, y[i], yp)

    def eval_and_learn(self, x, y):
        out = np.zeros(self.nrState)
        phiX = self.approximator[0].aggreg(x)
        for i in range(self.nrState):
            out[i] = self.approximator[i].evaluatePhiX(phiX)
            self.approximator[i].alpha += self.learner[i].learn(x, phiX, y[i], out[i])
        return out
        
    def reset(self):
        for i in range(self.nrState):
            self.approximator[i].reset()
            self.learner[i].reset()
        
    def __call__(self, x, u, index=0):

        x[2] = np.mod(x[2], 2 * np.pi)    # bearing angle must be in [0,2Pi]    
        self.fltr1.append(x)
        x2 = np.copy(x)
        x2[2] = x2[2] - 2 * np.pi if x2[2] > np.pi else x2[2]
        self.fltr2.append(x2)    # second filter array for transition from negative values to 2Pi
        x_fil = np.copy(x)
        
        if self.total_delay == self.delay + self.delayFIR:
            # FIR Smoothing
            ary1 = np.asarray(self.fltr1)
            ary2 = np.asarray(self.fltr2)
            if np.var(ary1[:,2]) < np.var(ary2[:,2]):    # variance for choosing range of bearing angle
                x_fil = self.fir(ary1)
            else:
                x_fil = self.fir(ary2)
            x_fil[2] = np.mod(x_fil[2], 2 * np.pi)    # bearing angle must be in [0,2Pi]    
            
            # Parameter estimation
            self.signal = self.ctrl[0]
            xLearn = np.hstack([self.measure[2], self.signal])
            yLearn = x_fil - self.measure
            yLearn[2] = np.mod(yLearn[2] + np.pi, 2 * np.pi) - np.pi
            self.learn(xLearn, yLearn)
            self.measure = x_fil
        else:
            self.total_delay += 1

        self.ctrl.append(u)
        self.output = x_fil
        
        # One step ahead state estimation
        for i in range(self.total_delay):
            self.output = self.output + self.evaluate(self.output[2], self.ctrl[i])
            self.output[2] = np.mod(self.output[2], 2 * np.pi)    # bearing angle must be in [0,2Pi]
        
        return self.output
