# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:57:40 2015

@author: JSCHOENK
"""
from __future__ import division
import numpy as np

from LearnSys.SparseILS import SparseILS as ILS

class TNTC:

    def __init__(self,foot):
        self.foot = foot
        self.output = np.zeros(1)
        self.name = foot["name"]
        
        self.control = ILS({"approximator":{"kind":"LIP" \
                                            ,"inputs":[{"kind":"GLTlinear","nodes":np.linspace(-np.pi/2,np.pi/2,11)} \
                                                      ,{"kind":"GLTlinear","nodes":np.linspace(-np.pi,np.pi,11)} \
                                                      ,{"kind":"GLTlinear","nodes":np.linspace(-np.pi/4,np.pi/4,11)} \
                                                      ]} \
                        ,"learner":{}})
        self.controlGamma = ILS({"approximator":{"kind":"LIP" \
                                            ,"inputs":[{"kind":"GLTlinear","nodes":np.linspace(-np.pi/4,np.pi/4,11)} \
                                                      ,{"kind":"GLTlinear","nodes":np.linspace(-np.pi/4,np.pi/4,11)} \
                                                      ]} \
                        ,"learner":{}})
        x_in = np.linspace(-np.pi/4,np.pi/4,11)
        for x in x_in:
            self.controlGamma.learn(np.array([np.pi/180*30,x]),np.pi/2)
            self.controlGamma.learn(np.array([-np.pi/180*30,x]),-np.pi/2)
        self.controlThetaS = ILS({"approximator":{"kind":"LIP" \
                                            ,"inputs":[{"kind":"GLTlinear","nodes":np.linspace(-np.pi/4,np.pi/4,11)} \
#                                                      ,{"kind":"GLTlinear","nodes":np.linspace(-np.pi/4,np.pi/4,11)} \
                                                      ]} \
                        ,"learner":{}})
                        
        
        
    def connectInputs(self,names):
        self.input = names[self.foot["input"]]
        self.r = self.input.r
        self.L_S = self.input.L_S
        self.L_C = self.input.L_C
        self.maxSteering = self.input.maxSteering
        self.e_theta_C_max = np.arcsin(self.r*np.sin(self.maxSteering)/(self.L_C+self.L_S))
        
        self.lastGamma = self.input.extra[2]
        self.lastOut = 0
        
        self.lastBeta = self.input.extra[1]
        self.lastGammaSoll = 0
        
        self.lastThetaS = self.input.state[2]
        self.lastThetaS_soll = 0
        
        
        
    def run(self,index):
        state = self.input.output
        extra = self.input.extra
        alpha = state[0]
        beta = extra[1]
        gamma = extra[2]
        theta_S = state[2]
        theta_C = state[3]
        x_eval = np.array([alpha,beta,gamma])
        max_gamma = np.pi/180*30
        
        
        theta_S_soll = 2*alpha
        
        
        # train last action - BETA
        self.controlThetaS.learn(np.array([theta_S-self.lastThetaS]),np.array(self.lastGamma))
        
        # eval current gamma_soll
        
        
        e_theta_S = np.mod(theta_S_soll - theta_S + np.pi,2*np.pi) - np.pi
        
        gamma_soll = self.controlThetaS(np.array([e_theta_S]))
        
        deltaGamma = e_theta_S/180*np.pi
        limit_gamma = 5
        if abs(deltaGamma)>limit_gamma:
            deltaGamma = limit_gamma*np.sign(deltaGamma)
#            
        gamma_learn = gamma_soll+deltaGamma
#        
        gamma_soll = self.controlThetaS.evalAndLearn(np.array([e_theta_S]),np.array(gamma_learn))
        
#        print "gamma: ",gamma,"deltaBeta: ",theta_S-self.lastThetaS
        
        

        if abs(gamma_soll)>max_gamma:
            gamma_soll = max_gamma*np.sign(gamma_soll)
            
            
#        gamma_soll = -np.pi/180*15
        

        out = self.controlGamma(np.array([gamma,gamma_soll]))
        
        e_gamma = np.mod(gamma_soll - gamma + np.pi,2*np.pi) - np.pi
        deltaOut = -e_gamma/np.pi*180
        limit = 10
        if abs(deltaOut)>limit:
            deltaOut = np.sign(deltaOut)*limit
        deltaOut *= np.pi/180
        
        
        # prevent jackknifing
        if abs(gamma)>max_gamma:
            out = np.pi/2*np.sign(gamma)
            deltaOut = 0
        
        # train last action - GAMMA
        self.controlGamma.learn(np.array([self.lastGamma,gamma]),np.array(self.lastOut))
        
        y_learn = out+deltaOut
        if gamma>np.pi/180*5 and y_learn<0:
            y_learn = 0
        if gamma<-np.pi/180*5 and y_learn>0:
            y_learn = 0
        
        # train according to hyper system - GAMMA
        out = self.controlGamma.evalAndLearn(np.array([gamma,gamma_soll]),y_learn)
        if gamma>np.pi/180*5 and out<0:
            out = 0
        if gamma<-np.pi/180*5 and out>0:
            out = 0
        
        # prevent jackknifing
        if abs(gamma)>max_gamma:
            out = np.pi/2*np.sign(gamma)
            
        # write output
        self.output = out
        
        # update 'last values'
        self.lastGamma = gamma
        self.lastOut = out
        
#        self.lastBeta = beta
#        self.lastGammaSoll = gamma_soll
        
        self.lastThetaS = theta_S
        self.lastThetaS_soll = theta_S_soll
        
        
    def end(self):
        0
        
    def reset(self):
        0