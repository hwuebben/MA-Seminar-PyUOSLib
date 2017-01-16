# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:57:40 2015

@author: JSCHOENK
"""
from __future__ import division
import numpy as np

from scipy.integrate import odeint
import pylab as plt

from BasicProcessingModule import BasicProcessingModule

# Coupled partial derivative equations
def f(state,time,param):
    # unpack parameter and input set:
    # Parameters:
    #   m - Point mass of the pendulum
    #   M - Mass of the cart
    #   l - Distance from the joint to the mass point m
    #   d_Mf - Viscous friction of the joint
    #   Fc - Coulomb friction coefficient
    #   g - gravitational acceleration
    # Input:
    #   u - external force on the cart
    m,M,l,d_Mf,Fc,g,u = param
    
    theta = state[0] # [rad]; Angular position of the pendulum
    dt_theta = state[1] # [rad/2]; Angular velocity of the pendulum
    dt_x = state[3] # [m/s]; Velocity of the cart

    a = 1/(m+M) # [1/kg]; reciprocial total mass
    f_c = Fc*np.sign(dt_x) # [N]; Coulomb friction between cart and track
    
    ddt_theta = (g * np.sin(theta) \
                 -m*l*a * dt_theta**2 * np.cos(theta) * np.sin(theta) \
                 -a*np.cos(theta) * (u-f_c) \
                 -(d_Mf*dt_theta)/(m*l)) \
                 /(2*l-m*l*a*np.cos(theta)**2)
    
    
    ddt_x = ( 2*a ) / ( 2 - m*a * np.cos(theta)**2 ) \
            * (m*l * dt_theta**2 * np.sin(theta) \
            - 1/2*m*g * np.cos(theta) * np.sin(theta) \
            + u \
            - f_c \
            + 1/(2*l) * np.cos(theta) * d_Mf * dt_theta)
    
    return [dt_theta,ddt_theta,dt_x,ddt_x]

class CartWithInvertedPendulum(BasicProcessingModule):

    def __init__(self,foot):
        BasicProcessingModule.__init__(self, foot)
        self.firstrun = True
        self.foot = foot
        self.output = []
        self.name = foot["ID"]
        try:
            self.dt = foot["dt"]
        except KeyError:
            self.dt = 0.01
        try:
            self.state = np.array(foot["state"])
        except KeyError:
            self.state = np.array([np.pi,0,0,0])
        self.m = 0.356 # [kg[; Point mass of the pendulum
        self.M = 4.8 # [kg]; Mass of the cart
        self.l = 0.56 # [m]; Distance from the joint to the mass point m
        self.dMF = 0.035 # [Nms/rad]; Viscous friction of the joint
        self.Fc = 4.9 # [N]; Coulomb friction coefficient
        self.g = 9.81 # [m/s^2]; "Gravitational constant"; gravitational acceleration (earth, mean, ...)
        self.L = 2 # [m]; Total length of tail
        self.Fmax = 120 # [N]; Maximum input value (actuator saturation)
        self.output = self.state.copy()
        
    def __call__(self, argIn, index=0):
        #test first run
        if self.firstrun:
            argIn = 10
            self.firstrun = False
        # actuator saturation
        if abs(argIn) > self.Fmax:
            argIn = self.Fmax*np.sign(argIn)
        # ODE solver parameters
        abserr = 1.0e-3
        relerr = 1.0e-2
        # Simulation parameters and inputs        
        p = [self.m,self.M,self.l,self.dMF,self.Fc,self.g,argIn]
        # ODE solver, one step: dt
        result = odeint(f, self.state, [0,self.dt], args=(p,), atol=abserr, rtol=relerr, printmessg=False)
        # update state
        self.state = result[1]
        return result[1]
        
    def end(self):
        # nothing to do here
        0
        
    def reset(self):
        # someday rest state here
        0
        
if __name__=='__main__':
    sim = CartWithInvertedPendulum({"name":"Balance"})
    state = []
    nrD = 10000
    for i in range(nrD):
        state.append(sim(120,i))
    plt.plot(range(nrD),state)
    plt.show()
