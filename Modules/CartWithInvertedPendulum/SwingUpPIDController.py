# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 11:22:50 2016

@author: joschnei
"""

from __future__ import division
import numpy as np

class SwingUpPIDController:
    def __init__(self, foot):
        self.foot = foot
        self.name = foot["name"]
        self.cnt = 0
        try:
            self.maxForce = foot["maxForce"]
        except(KeyError):
            self.maxForce = 120             # 120 N is the standard F_max in "Benchmark problems for nonlinear system identification and control using Soft Computing methods: Need and overview
        self.output = 0
        self.errorInt = 0
        self.errorOld = 0
        # Parameters tuned after Ziegler-Nichols
        # K_u = 1400
        # T_u = 80
        self.P = 250
        self.I = 0
        self.D = 0

    def connectInputs(self,names):
        # connect input for external force on the cart
        self.input = names[self.foot["input"]]
        
    def run(self,index):
        # read input
        argIn = self.input.output
        # perform simulation
        result = self(argIn,index)
        # provide output
        self.output = result

    def __call__(self, argIn, index):
        m = 0.356
        l = 0.56
        g = 9.81
        theta = np.fmod(argIn[0],np.pi)
        dt_theta = argIn[1]
        potEnergy = m*g*l*(2-((np.cos(theta)+1)))
        kineticEnergy = 0.5*m*(dt_theta*l)**2
        energy = kineticEnergy+potEnergy
        error = energy - m*g*l*2

        # too much energy -> reduce energy
        if(error < 0):
            result = (-1)*np.sign(argIn[2])*np.absolute(error)*self.P
        else:
            result = np.sign(argIn[2])*np.absolute(error)*self.P            
        if(result > 0.1 and result < 6):
            result += 6
        elif(result < -0.1 and result > -6):
            result -= 6
        return result
    
    def end(self):
        pass