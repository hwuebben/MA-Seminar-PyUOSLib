# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:57:40 2015

@author: JSCHOENK
"""
from __future__ import division

import numpy as np
import pylab as plt
from collections import deque

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

try:
    from MPyUOSLib import BasicProcessingModule
except ImportError:
    class BasicProcessingModule:
        def __init__(self, foot):
            pass


class PlotBicycleModel(BasicProcessingModule):
    """Localization of car on a trajectory based on pose (x,y,phi)
    
    x - Position on x-axis
    y - Position on y-axis
    phi - Orientation
    
    This module localizes a car on a trajectory given by its pose (x,y,phi) and
    calculate the distance and misaligment to the trajectory.
    """
    def __init__(self, foot):
        BasicProcessingModule.__init__(self, foot)
        
        try:
            trajLength = foot["trackLength"]
        except KeyError:
            trajLength = 1000
        
        self.pose = deque(maxlen=trajLength)
        self.est = deque(maxlen=trajLength)
#        self.error = deque(maxlen=trajLength)
        self.time = deque(maxlen=trajLength)
        
    def __call__(self, index=0, **argIn):
        try:
            self.pose.append(argIn["car"].copy())
        except KeyError:
            pass
        try:
            self.est.append(argIn["estimate"].copy())
        except KeyError:
            pass
#        try:
#            self.error.append(argIn["track"].copy())
#            t = self.inputArguments["track"].timeEstimation
#            self.time.append(self.inputArguments["track"].evalTrajectory(t))
#        except KeyError:
#            pass
    
    def end(self):
        plt.figure()
        trajPoints = []
        time = np.linspace(0, 1, 3601)
        for t in time:
            trajPoints.append(self.inputArguments["track"].evalTrajectory(t))
        trajPoints = np.array(trajPoints).T
        trajMarker = self.inputArguments["track"].points.T
        plt.plot(trajPoints[0,:], trajPoints[1,:])
        plt.plot(trajMarker[0,:], trajMarker[1,:], 'ko')
        position = np.array(self.pose).T
        estimation = np.array(self.est).T
#        error = np.array(self.error).T
        est = np.array(self.time).T
#        error_x = position[0,:]+np.cos(position[2,:]+error[1,:]-np.pi/2)*error[0,:]
#        error_y = position[1,:]+np.sin(position[2,:]+error[1,:]-np.pi/2)*error[0,:]
#        for i in range(len(self.pose)):
#            x = [position[0,i],est[0,i]]
#            y = [position[1,i],est[1,i]]
#            plt.plot(x, y)
#        for i in range(len(self.pose_est)):
#            x = [estimation[0,i],est[0,i]]
#            y = [estimation[1,i],est[1,i]]
#            plt.plot(x, y)
        
        plt.plot(position[0,:], position[1,:], 'r.')
        if estimation != []:
            plt.plot(estimation[0,:], estimation[1,:], 'g.')
#        plt.plot(est[0,:], est[1,:], 'g.')
        
#        plt.plot(error_x,error_y,'k.')
        plt.axis('equal')
        plt.show()


class PlotLearningHistory(BasicProcessingModule):
    """Save and plot learning history of an IncrementalLearningSystem module
    
    """
    def __init__(self, foot):
        BasicProcessingModule.__init__(self, foot)
        
        self.sampling_domain = foot["samplingDomain"]
        self.sampling_size = foot["samplingSize"]
        
        try:
            self.srange = foot["samplingRange"]
        except KeyError:
            self.srange = 10
        try:
            self.outDim = foot["outputDim"]
        except KeyError:
            self.outDim = 0
        
        x = np.linspace(*self.sampling_domain[0], num=self.sampling_size)
        y = np.linspace(*self.sampling_domain[1], num=self.sampling_size)
        xv, yv = np.meshgrid(x, y)
        
        self.xDim = xv
        self.yDim = yv
        
        self.approximation = []
        self.xLearn = []
        self.yLearn = []
        
        self.measure = deque(maxlen=foot["delay"])
        self.measure.append(np.array([0,0,1]))
        self.nrD = 0
    
    def prepare(self, antecessor):
        learning_system = antecessor["ILS"]
        
        yPred = np.zeros(np.shape(self.xDim) + (3,))
        for i in xrange(self.sampling_size):
            for j in xrange(self.sampling_size):
                yPred[i,j] = learning_system.evaluate(self.xDim[i,j], self.yDim[i,j])
        
        self.approximation.append(yPred[:,:,self.outDim])
    
    def __call__(self, index=0, **argIn):
        if self.nrD % self.srange == 0:
            learning_system = self.inputArguments["ILS"]
            
            yPred = np.zeros(np.shape(self.xDim) + (3,))
            for i in xrange(self.sampling_size):
                for j in xrange(self.sampling_size):
                    yPred[i,j] = learning_system.evaluate(self.xDim[i,j], self.yDim[i,j])
            
            self.approximation.append(yPred[:,:,self.outDim])
            self.xLearn.append(self.measure[0])
            self.yLearn.append(argIn["xLearn"] - self.measure[0])
            
        self.measure.append(argIn["xLearn"])   
        self.nrD += 1
    
    def end(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        self.mesh = ax.plot_wireframe(self.xDim, 
                                      self.yDim, 
                                      self.approximation[-1])
        self.last_date = ax.scatter(self.xLearn[-1][0], 
                                    self.xLearn[-1][1], 
                                    self.yLearn[-1][self.outDim], 
                                    c='r')
        self.current_date = ax.scatter(self.xLearn[-1][0], 
                                       self.xLearn[-1][1], 
                                       self.yLearn[-1][self.outDim], 
                                       c='g')
        
        axcolor = 'white'
        axfreq = plt.axes([0.1, 0.01, 0.85, 0.03], axisbg=axcolor)
        sfreq = Slider(axfreq, 'Sample', 0, len(self.yLearn), valinit=len(self.yLearn))
        
        def update2D(val):
            sample_index = int(round(sfreq.val))
            last_index = max([sample_index-1,0])
            current_index = min([sample_index,len(self.yLearn)-1])
            
            self.mesh.remove()
            self.last_date.remove()
            self.current_date.remove()
            
            self.mesh = ax.plot_wireframe(self.xDim,
                                          self.yDim, 
                                          self.approximation[sample_index])
            self.last_date = ax.scatter(self.xLearn[last_index][0], 
                                        self.xLearn[last_index][1],
                                        self.yLearn[last_index][self.outDim], 
                                        c='r')
            self.current_date = ax.scatter(self.xLearn[current_index][0],
                                           self.xLearn[current_index][1],
                                           self.yLearn[current_index][self.outDim], 
                                           c='g')
            
            plt.draw()
        
        ax.set_xlabel('Theta')
        ax.set_ylabel('Phi')
        ax.set_zlim(-1, 1)
        
        sfreq.on_changed(update2D)
        plt.ion()
        plt.show()
