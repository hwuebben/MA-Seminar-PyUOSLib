# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:57:40 2015

@author: JSCHOENK
"""
from __future__ import division
import numpy as np

from scipy.integrate import odeint
import pylab as plt

# Coupled partial derivative equations
def f(state,time,param):
    # unpack parameter and input set:
    # Parameters:
    #   M - Mass of the cart
    #   m - Point mass of the pendulum
    #   b - Friction of the cart    
    #   I - Inertia of the pendulum
    #   l - Distance from the joint to the mass point m
    # Input:
    #   F - external force on the cart
    M,m,b,I,l,F = param
    
    theta = state[0] # [rad]; Angular position of the pendulum
    dt_theta = state[1] # [rad/2]; Angular velocity of the pendulum
    dt_x = state[3] # [m/s]; Velocity of the cart

#    norm = I*(M+m)+M*m*l**2
    g = 9.81
    
    ddt_x = (F+m*l*dt_theta**2*np.sin(theta)-m*g*np.cos(theta)*np.sin(theta))/(M+m-m*(np.cos(theta))**2)
    
    ddt_theta = (F*np.cos(theta)-(M+m)*g*np.sin(theta)+m*l*dt_theta*np.cos(theta)*np.sin(theta))/(m*l*(np.cos(theta))**2-(M+m)*l)
            
    return [dt_x,ddt_x,dt_theta,ddt_theta]

class CartWithInvertedPendulumEasy:

    def __init__(self,foot):
        self.foot = foot
        self.output = []
        self.name = foot["name"]
        try:
            self.dt = foot["dt"]
        except KeyError:
            self.dt = 0.01
        try:
            self.state = np.array(foot["state"])
        except KeyError:
            self.state = np.array([np.pi,0,0,0])
        
        self.M = 0.5 # [kg]; Mass of the cart
        self.m = 0.2 # [kg[; Point mass of the pendulum
        self.b = 0.1 # [N/m/s]; Friction of the cart
        self.I = 0.006 # [kg*m^2] ; Inertia of the pendulum
        self.l = 0.3 # [m]; Distance from the joint to the mass point m
        self.Fmax = 120 # [N]; Maximum input value (actuator saturation)
        
    def connectInputs(self,names):
        # connect input for external force on the cart
        self.input = [names[self.foot["input"]]]
        
    def run(self,index):
        # read input
        argIn = self.input.output
        # perform simulation
        result = self(argIn,index)
        # provide output
        self.output = result
        
    def __call__(self,argIn,index):
        # actuator saturation
        if abs(argIn) > self.Fmax:
            argIn = self.Fmax*np.sign(argIn)
        # ODE solver parameters
        abserr = 1.0e-10
        relerr = 1.0e-8
        # Simulation parameters and inputs        
        p = [self.M,self.m,self.b,self.I,self.l,argIn]
        # ODE solver, one step: dt
        result = odeint(f, self.state, [0,self.dt], args=(p,), atol=abserr, rtol=relerr, mxstep=500)
        # update state
        self.state = result[1]
        self.state[0] = np.mod(self.state[0]+np.pi,2*np.pi)-np.pi
        return result[1]
        
    def end(self):
        # nothing to do here
        0
        
    def reset(self):
        # someday rest state here
        0
        
if __name__=='__main__':
    sim = CartWithInvertedPendulumEasy({"name":"Balance","state":[3.14159,0,0,0.1],"dt":0.01})
    state = []
    nrD = 1000
    for i in range(nrD):
        state.append(sim(000,i))
    plt.plot(range(nrD),state)
    plt.legend(['theta','dtheta','x','dx'])
    plt.show()