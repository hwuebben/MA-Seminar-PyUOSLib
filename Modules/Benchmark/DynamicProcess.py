# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:57:40 2015

@author: JSCHOENK
"""
from __future__ import division
import numpy as np
from scipy.integrate import odeint

from MPyUOSLib import BasicProcessingModule

"""
This module groups different dynamic processess for modelling and control
"""

class Example1(BasicProcessingModule):
    """
    Example 1 
    
    from:
    
    Narendra, Kumpati S., and Kannan Parthasarathy. 
    "Identification and control of dynamical systems using neural networks." 
    Neural Networks, IEEE Transactions on 1.1 (1990): 4-27.
    
    Signal for identification: U([-1,1])
    """
    def __init__(self, foot):
        default = {"a0":0.3,"a1":0.6}
        BasicProcessingModule.__init__(self, foot, default)
        self.output = np.zeros(1)
        self.y_k = 0
        self.y_k_1 = 0
        
    def __call__(self, u, a0=None, a1=None, index=0):
        if a0 is None:
            a0 = self.a0
        if a1 is None:
            a1 = self.a0
        f = 0.6*np.sin(np.pi*u) + 0.3*np.sin(3*np.pi*u)+ 0.1*np.sin(5*np.pi*u)
        next_y = a0*self.y_k + a1*self.y_k_1 + f
        self.y_k_1 = self.y_k
        self.y_k = next_y
        return self.y_k

class Example2(BasicProcessingModule):
    """
    Example 2 
    
    from:
    
    Narendra, Kumpati S., and Kannan Parthasarathy. 
    "Identification and control of dynamical systems using neural networks." 
    Neural Networks, IEEE Transactions on 1.1 (1990): 4-27.
    
    Signal for identification: U([-2,2])
    """
    def __init__(self, foot):
        default = {"a0":2.5}
        BasicProcessingModule.__init__(self, foot, default)
        self.output = np.zeros(1)
        self.y_k = 0
        self.y_k_1 = 0
        
    def __call__(self, u, a0=None, index=0):
        if a0 is None:
            a0 = self.a0
        next_y = (self.y_k*self.y_k_1*(self.y_k+a0))/(1+self.y_k**2*self.y_k_1**2) + u
        self.y_k_1 = self.y_k
        self.y_k = next_y
        return self.y_k
        
class Example3(BasicProcessingModule):
    """
    Example 3 
    
    from:
    
    Narendra, Kumpati S., and Kannan Parthasarathy. 
    "Identification and control of dynamical systems using neural networks." 
    Neural Networks, IEEE Transactions on 1.1 (1990): 4-27.
    
    Signal for identification: U([-2,2])
    """
    def __init__(self, foot):
        default = {"a0":1.0}
        BasicProcessingModule.__init__(self, foot, default)
        self.output = np.zeros(1)
        self.y_k = 0
        
    def __call__(self, u, a0=None, index=0):
        if a0 is None:
            a0 = self.a0
        next_y = (a0*self.y_k)/(1+self.y_k**3) + u**3
        self.y_k = next_y
        return self.y_k
        
class Example4(BasicProcessingModule):
    """
    Example 4 
    
    from:
    
    Narendra, Kumpati S., and Kannan Parthasarathy. 
    "Identification and control of dynamical systems using neural networks." 
    Neural Networks, IEEE Transactions on 1.1 (1990): 4-27.
    
    Signal for identification: U([-1,1])
    """
    def __init__(self, foot):
        default = {"a0":1.0}
        BasicProcessingModule.__init__(self, foot, default)
        self.output = np.zeros(1)
        self.y_k = 0
        self.y_k_1 = 0
        self.y_k_2 = 0
        self.u_k_1 = 0
        
    def __call__(self, u, a0=None, index=0):
        if a0 is None:
            a0 = self.a0
        x_1 = self.y_k
        x_2 = self.y_k_1
        x_3 = self.y_k_2
        x_4 = u
        x_5 = self.u_k_1
        
        next_y = (x_1*x_2*x_3*x_5*(x_3-a0)+x_4)/(1+x_3**2+x_2**2)
        self.y_k_2 = self.y_k_1
        self.y_k_1 = self.y_k
        self.y_k = next_y
        self.u_k_1 = u
        return self.y_k

class Example5(BasicProcessingModule):
    """
    Example 5 
    
    from:
    
    Narendra, Kumpati S., and Kannan Parthasarathy. 
    "Identification and control of dynamical systems using neural networks." 
    Neural Networks, IEEE Transactions on 1.1 (1990): 4-27.
    
    Signal for identification: U([-1,1]),U([-1,1])
    """
    def __init__(self, foot):
        default = {"a0":1.0}
        BasicProcessingModule.__init__(self, foot, default)
        self.output = np.zeros(2)
        self.y_k = np.zeros(2)
        
    def __call__(self, u_1, u_2, a0=None, index=0):
        if a0 is None:
            a0 = self.a0
        next_y_1 = (self.y_k[0])               / (a0 + self.y_k[1]**2) + u_1
        next_y_2 = (self.y_k[0] * self.y_k[0]) / (a0 + self.y_k[1]**2) + u_2
        self.y_k = np.array([next_y_1,next_y_2])
        return self.y_k
        
class Example6(BasicProcessingModule):
    """
    Example 6 
    
    from:
    
    Narendra, Kumpati S., and Kannan Parthasarathy. 
    "Identification and control of dynamical systems using neural networks." 
    Neural Networks, IEEE Transactions on 1.1 (1990): 4-27.
    
    Signal for identification: U([-1,1])
    """
    def __init__(self, foot):
        default = {"a0":0.8}
        BasicProcessingModule.__init__(self, foot, default)
        self.output = np.zeros(1)
        self.y_k = 0
        
    def __call__(self, u, a0=None, index=0):
        if a0 is None:
            a0 = self.a0
        f = (u-0.8)*u*(u+0.5)
        next_y = a0*self.y_k + f
        self.y_k = next_y
        return self.y_k

       
class PendulumSwingUp(BasicProcessingModule):
    """
    Pendulum Swing-Up Task
    
    from:
    
    Eerland, Willem, Coen de Visser, and Erik-Jan van Kampen. 
    "On Approximate Dynamic Programming with Multivariate Splines for 
    Adaptive Control." arXiv preprint arXiv:1606.09383 (2016).
    
    limited input torque:
    T^max = 5
    """
    
    def __init__(self, foot):
        BasicProcessingModule.__init__(self,foot)
        if not hasattr(self,"state"):
            self.state = np.array([np.pi,0.0])
        else:
            self.state = np.array(self.state)
        self.iniState = self.state.copy()
        self.output = self.state.copy()
        self.firstrun = True
        if not hasattr(self,"Tmax"):
            self.Tmax = 5.0 # [Nm]
        if not hasattr(self,"m"):
            self.m = 1.0 # [kg]
        if not hasattr(self,"l"):
            self.l = 1.0 # [m]
        if not hasattr(self,"mu"):
            self.mu = 0.01 # [1]
        if not hasattr(self,"g"):
            self.g = 9.81 # [m/s^2]
        if not hasattr(self,"sigma_w"):
            self.sigma_w = 0.0 # [°/s^2]
        if not hasattr(self,"dt"):
            self.dt = 0.1 # [s]
            
        
    def f(self, state, time, param):
        # unpack parameter and input set:
        # Parameters:
        #   m - Point mass of the pendulum
        #   l - Distance from the joint to the mass point m
        #   mu - Viscous friction of the joint
        #   g - gravitational acceleration
        # Input:
        #   u - external force on the cart
        # Process noise:
        #   sigma_w - standard deviation of white process noise
        m,l,mu,g,u, sigma_w = param
        
        theta = state[0] # [rad]; Angular position of the pendulum
        dt_theta = state[1] # [rad/2]; Angular velocity of the pendulum
    
        if sigma_w > 0.0:
            ddt_theta = g/l*np.sin(theta)-mu/(m*l**2)*dt_theta+1/(m*l**2)*u+np.random.normal(scale=sigma_w)
        else:
            ddt_theta = g/l*np.sin(theta)-mu/(m*l**2)*dt_theta+1/(m*l**2)*u
        return [dt_theta,ddt_theta]
        
    def __call__(self, argIn, index=0, m=None, l=None, mu=None, g=None, sigma_w=None):
        if m is None:
            m = self.m
        if l is None:
            l = self.l
        if mu is None:
            mu = self.mu
        if g is None:
            g = self.g
        if sigma_w is None:
            sigma_w = self.sigma_w
        
        #test first run
        if self.firstrun:
            argIn = self.Tmax
            self.firstrun = False
        # actuator saturation
        if abs(argIn) > self.Tmax:
            argIn = self.Tmax*np.sign(argIn)
        # ODE solver parameters
        abserr = 1.0e-3
        relerr = 1.0e-2
        # Simulation parameters and inputs        
        p = [m, l, mu, g, argIn, sigma_w]
        # ODE solver, one step: dt
        result = odeint(self.f, self.state, [0,self.dt], args=(p,), atol=abserr, rtol=relerr, printmessg=False)
        # update state
        self.state = result[1]
        self.state[0] = np.mod(self.state[0]+np.pi,2*np.pi)-np.pi
        return np.array(self.state)
        
    def reset(self):
        self.state = self.iniState.copy()
    
    def randReset(self):
        phi = np.random.uniform(-np.pi,np.pi)
        self.state = np.array([phi,0.0])
        
class SISOSin(BasicProcessingModule):
    def __init__(self, foot):
        default = {"state":[1]}
        BasicProcessingModule.__init__(self, foot, default)
        self.state = np.array(self.state)
        self.iniState = self.state.copy()
        self.output = self.state.copy()

            
        
    def f(self, state, time, param):
        # unpack parameter and input set:
        # Parameters:
        #   m - Point mass of the pendulum
        #   l - Distance from the joint to the mass point m
        #   mu - Viscous friction of the joint
        #   g - gravitational acceleration
        # Input:
        #   u - external force on the cart
        # Process noise:
        #   sigma_w - standard deviation of white process noise
        
        return np.sin(state)+np.array(param)
        
    def __call__(self, argIn, index=0):
        
        # ODE solver parameters
        abserr = 1.0e-3
        relerr = 1.0e-2
        # Simulation parameters and inputs        
        p = argIn
        # ODE solver, one step: dt
        result = odeint(self.f, self.state, [0,self.dt], args=(p,), atol=abserr, rtol=relerr, printmessg=False)
        # update state
        self.state = result[1]
        return np.array(self.state)
        
    def reset(self):
        self.state = self.iniState.copy()
    
    def randReset(self):
        self.state = np.random.uniform(-np.pi,np.pi)