# -*- coding: utf-8 -*-
"""
Created on Fri Mar 04 12:51:58 2016

@author: JSCHOENK
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from MPyUOSLib import BasicProcessingModule

"""
This module groups different plotting functions
"""

class Plot(BasicProcessingModule):
    """
    Basic plotting module
    """
    def __init__(self, foot):
        default = {"save" : None, "format":'pdf', "savePath":'', "title":''}
        BasicProcessingModule.__init__(self, foot, default)
        self.reset()
        
    def reset(self):
        self.dataX = []
        self.dataY = []
        
    
    def prepare(self, antecessor):
        try:
            self.dataX.append(antecessor["x"].output)
        except KeyError:
            pass
        
        self.dataY.append(antecessor["y"].output)
        self.nrY = len(antecessor["y"].output)
        if antecessor.has_key("save"):
            self.save = antecessor["save"].output
            if not ('.' in self.save):
                self.save = self.save + '.' + self.format
            
        
    
    def __call__(self, index=0, save=None, **argIn):
        try:
            self.dataX.append(argIn["x"])
        except KeyError:
            pass
  
        self.dataY.append(argIn["y"])
        
    def end(self):        
        self.dataY = np.array(self.dataY)
        self.figure = plt.figure()
        if self.dataX:
            self.dataX = np.array(self.dataX)
            plt.plot(self.dataX,self.dataY)
        else:
            plt.plot(range(len(self.dataY)),self.dataY)
        try:
            plt.xlabel(self.xlabel)
        except AttributeError:
            pass
        try:
            plt.ylabel(self.ylabel)
        except AttributeError:
            pass
        try:
            plt.legend(self.legend)
        except AttributeError:
            pass
        
        if self.save is None:
            plt.show()
        else:
            self.figure.savefig(self.savePath + self.save, format=self.format)
            plt.close()
        
class PlotVsTime(BasicProcessingModule):
    """
    Basic plotting module
    """
    def __init__(self,foot):
        default = {"save":None, "format":'pdf', "savePath":'', "title":''}
        BasicProcessingModule.__init__(self, foot, default)
        self.reset()
        
    def reset(self):
        self.data = {}
    
    def prepare(self, antecessor):
        for k,v in antecessor.iteritems():
            if not (k == 'save'):
                self.data[k] = [v.output]
            
        if antecessor.has_key("save"):
            self.save = antecessor["save"].output
            if not ('.' in self.save):
                self.save = self.save + '.' + self.format
            
    
    def __call__(self, index=0, save=None, **argIn):
        for k,v in argIn.iteritems():
            self.data[k].append(v)
        
    def end(self):
        self.figure = plt.figure()
        for k,v in self.data.iteritems():
            plt.plot(v)
        plt.legend(self.data.keys())
        try:
            plt.xlabel(self.foot["label"][0])
            plt.ylabel(self.foot["label"][1])
        except KeyError:
            pass
        
        if self.save is None:
            plt.show()
        else:
            self.figure.savefig(self.savePath + self.save, format=self.format)
            plt.close()
        
class PlotScatter3D(BasicProcessingModule):
    def __init__(self,foot):
        default = {"save" : None, "format":'pdf', "savePath":'', "title":'', "sikpInit":False, "zlim":False}
        BasicProcessingModule.__init__(self, foot, default)
        self.reset()
        
    def reset(self):
        self.x = []
        self.y = []
    
    def prepare(self, antecessor):
        if not self.skipInit:
            self.x.append(antecessor["x"].output)
            self.y.append(antecessor["y"].output)
            
        if antecessor.has_key("save"):
            self.save = antecessor["save"].output
            if not ('.' in self.save):
                self.save = self.save + '.' + self.format
        
    def __call__(self, x, y, index=0):
        self.x.append(x)
        self.y.append(y)
        
    def end(self):
        x = np.array(self.x)
        y = np.array(self.y)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.ax.scatter(x[:,0], x[:,1], y, c=range(len(y)))
        
        self.ax.set_xlabel('X[0]')
        self.ax.set_ylabel('X[1]')
        self.ax.set_zlabel('Y')
        if self.zlim:
            self.ax.set_zlim(self.zlim)
        if self.save is None:
            plt.show()
        else:
            self.fig.savefig(self.savePath + self.save, format=self.format)
            plt.close()
        
class PlotFinalApproximation(BasicProcessingModule):
    """Plot the final approximation 
    of an IncrementalLearningSystem 
    module after learning
    
    """
    def __init__(self,foot):
        BasicProcessingModule.__init__(self,foot)
        if np.size(self.samplingDomain) == 2:
            self.inputDim = 1
        elif np.size(self.samplingDomain) == 4:
            self.inputDim = 2

    def reset(self):
        pass
            
    def prepare(self,antecessor):
        pass
        
    def __call__(self, index=0, **argIn):
        pass
        
    def end(self):
        observed_learning_system = self.inputArguments["ILS"]
        if self.inputDim == 1:
            x = np.linspace(*self.samplingDomain,num=self.samplingSize)
            yPred = np.zeros(self.samplingSize)
            for i in xrange(self.samplingSize):
                yPred[i] = observed_learning_system.evaluate(x[i])
            plt.figure()
            plt.plot(x,yPred)
        elif self.inputDim == 2:
            dimSize = int(np.floor(np.sqrt(self.samplingSize)))
            x = np.linspace(*self.samplingDomain[0],num=dimSize)
            y = np.linspace(*self.samplingDomain[1],num=dimSize)
            xv,yv = np.meshgrid(x,y)
            self.samplingSize = dimSize**2
            
            yPred = np.zeros(np.shape(xv))
            for i in xrange(dimSize):
                for j in xrange(dimSize):
                    yPred[i,j] = observed_learning_system.evaluate(np.array([xv[i,j],yv[i,j]]))
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(xv, yv, yPred, rstride=1, cstride=1, 
                            cmap=cm.coolwarm, linewidth=0, antialiased=False)
            plt.show()
        else:
            pass
        
        
class PlotLearningHistory(BasicProcessingModule):
    """Save & plot the learning history of an IncrementalLearningSystem module
    
    """
    def __init__(self,foot):
        BasicProcessingModule.__init__(self,foot)
        if np.size(self.samplingDomain) == 2:
            self.inputDim = 1
            self.samplingPoints = np.linspace(*self.samplingDomain,num=self.samplingSize)
            self.dimSize = self.samplingSize
        elif np.size(self.samplingDomain) == 4:
            self.inputDim = 2
            dimSize = int(np.floor(np.sqrt(self.samplingSize)))
            x = np.linspace(*self.samplingDomain[0],num=dimSize)
            y = np.linspace(*self.samplingDomain[1],num=dimSize)
            xv,yv = np.meshgrid(x,y)
            self.samplingSize = dimSize**2
            self.dimSize = dimSize
            self.x_dim = xv
            self.y_dim = yv

        self.approximation = []
        self.xLearn = []
        self.yLearn = []
        
    def reset(self):
        self.approximation = []
        self.xLearn = []
        self.yLearn = []
            
    def prepare(self,antecessor):
        observed_learning_system = antecessor["ILS"]
        
        if self.inputDim == 1:
            yPred = np.zeros(self.samplingSize)
            for i in np.arange(self.samplingSize):
                yPred[i] = observed_learning_system.evaluate(self.samplingPoints[i])
        elif self.inputDim == 2:
            yPred = np.zeros(np.shape(self.x_dim))
            for i in xrange(self.dimSize):
                for j in xrange(self.dimSize):
                    yPred[i,j] = observed_learning_system.evaluate(np.array([self.x_dim[i,j],self.y_dim[i,j]]))
        else:
            pass
            
        self.approximation.append(yPred)
        
    def __call__(self, index=0, **argIn):
        observed_learning_system = self.inputArguments["ILS"]
        if self.inputDim == 1:
            yPred = np.zeros(self.samplingSize)
            for i in np.arange(self.samplingSize):
                yPred[i] = observed_learning_system.evaluate(self.samplingPoints[i])
        elif self.inputDim == 2:
            yPred = np.zeros(np.shape(self.x_dim))
            for i in xrange(self.dimSize):
                for j in xrange(self.dimSize):
                    yPred[i,j] = observed_learning_system.evaluate(np.array([self.x_dim[i,j],self.y_dim[i,j]]))
        else:
            pass
            
        self.approximation.append(np.array(yPred))
        self.xLearn.append(np.array(argIn["xLearn"]))
        self.yLearn.append(np.array(argIn["yLearn"]))        
        
    def end(self):
        gtAlpha = 0.5
        if self.inputDim == 1:
            if self.inputArguments.has_key("GroundTruth"):
                groundTruthModule = self.inputArguments["GroundTruth"]
                keyList = groundTruthModule.foot["input"].keys()
                if "modules" in keyList:
                    keyList.remove("modules")
                key = keyList[0]
                fig = plt.figure()
                axesMain = plt.axes([0.1,0.07,0.85,0.8])
                L1, = axesMain.plot(self.samplingPoints,self.approximation[-1])
                L2, = axesMain.plot(self.samplingPoints,self.approximation[-2],'--')
                L3, = axesMain.plot(self.xLearn[-1],self.yLearn[-1],'or')
                yGT = np.zeros(self.samplingSize)
                for i in np.arange(self.samplingSize):
                    yGT[i] = groundTruthModule(**{key:self.samplingPoints[i],"index":len(self.yLearn)})
                L4, = axesMain.plot(self.samplingPoints,yGT,'-.')
                
                axcolor = 'white'
                axfreq = plt.axes([0.1, 0.01, 0.85, 0.03], axisbg=axcolor)
                sfreq = Slider(axfreq, '#Sample', 0, len(self.yLearn)-1, valinit=len(self.yLearn)-1)
                def update1D(val):
                    sample_index = int(np.floor(sfreq.val))
                    L1.set_ydata(self.approximation[sample_index+1])
                    L2.set_ydata(self.approximation[sample_index])
                    L3.set_xdata(self.xLearn[sample_index])
                    L3.set_ydata(self.yLearn[sample_index])
                    yGT = np.zeros(self.samplingSize)
                    for i in np.arange(self.samplingSize):
                        yGT[i] = groundTruthModule(**{key:self.samplingPoints[i],"index":sample_index})
                    L4.set_ydata(yGT)
                    
                ymin = min(self.yLearn)
                ymax = max(self.yLearn)
                yrange = ymax-ymin
                ymin -= 0.1*yrange
                ymax += 0.1*yrange
                axesMain.set_ylim([ymin,ymax])
#                axesMain.xlim([])
                sfreq.on_changed(update1D)
                plt.show()
            else:
                fig = plt.figure()
                axesMain = plt.axes([0.1,0.07,0.85,0.8])
                L1, = axesMain.plot(self.samplingPoints,self.approximation[-1])
                L2, = axesMain.plot(self.samplingPoints,self.approximation[-2],'--')
                L3, = axesMain.plot(self.xLearn[-1],self.yLearn[-1],'or')
                
                axcolor = 'white'
                self.axfreq = plt.axes([0.1, 0.01, 0.85, 0.03], axisbg=axcolor)
                self.sfreq = Slider(self.axfreq, '#Sample', 0, len(self.yLearn)-1, valinit=len(self.yLearn)-1)
                def update1D(val):
                    sample_index = int(round(sfreq.val))
                    L1.set_ydata(self.approximation[sample_index+1])
                    L2.set_ydata(self.approximation[sample_index])
                    L3.set_xdata(self.xLearn[sample_index])
                    L3.set_ydata(self.yLearn[sample_index])
                self.sfreq.on_changed(update1D)
                plt.show()
        elif self.inputDim == 2:
            if self.inputArguments.has_key("GroundTruth"):
                groundTruthModule = self.inputArguments["GroundTruth"]
                key = groundTruthModule.foot["input"].keys()[0]
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                self.surf = ax.plot_surface(self.x_dim, 
                                        self.y_dim, 
                                        self.approximation[-1], 
                                        rstride=1, 
                                        cstride=1, 
                                        cmap=cm.coolwarm,
                                        linewidth=0, 
                                        antialiased=False,alpha=0.9)
                                        
                self.lastLearningDate = ax.scatter(self.xLearn[-1][0], 
                                                   self.xLearn[-1][1], 
                                                   self.yLearn[-1])
                                                   
                self.currentLearningDate = ax.scatter(self.xLearn[-1][0], 
                                                      self.xLearn[-1][1], 
                                                      self.yLearn[-1])
                                                      
                yGT = np.zeros(np.shape(self.x_dim))
                for i in np.arange(self.dimSize):
                    for j in np.arange(self.dimSize):
                        yGT[i,j] = groundTruthModule(**{key:np.array([self.x_dim[i,j],self.y_dim[i,j]]),"index":len(self.yLearn)})
                self.gtScatter = ax.scatter(self.x_dim.flatten(),self.y_dim.flatten(),yGT.flatten(),c=yGT.flatten(),alpha=gtAlpha)

                axcolor = 'white'
                self.axfreq = plt.axes([0.1, 0.01, 0.85, 0.03], axisbg=axcolor)
                self.sfreq = Slider(self.axfreq, '#Sample', 0, len(self.yLearn), valinit=len(self.yLearn))
                def update2D(val):
                    sample_index = int(round(self.sfreq.val))
                    self.surf.remove()
                    self.lastLearningDate.remove()
                    self.currentLearningDate.remove()
                    self.gtScatter.remove()
                    yGT = np.zeros(np.shape(self.x_dim))
                    for i in np.arange(self.dimSize):
                        for j in np.arange(self.dimSize):
                            yGT[i,j] = groundTruthModule(**{key:np.array([self.x_dim[i,j],self.y_dim[i,j]]),"index":len(self.yLearn)})
                    self.gtScatter = ax.scatter(self.x_dim.flatten(),self.y_dim.flatten(),yGT.flatten(),c=yGT.flatten(),alpha=gtAlpha)
                    
                    self.surf = ax.plot_surface(self.x_dim, 
                                        self.y_dim, 
                                        self.approximation[sample_index], 
                                        rstride=1, 
                                        cstride=1, 
                                        cmap=cm.coolwarm,
                                        linewidth=0, antialiased=False,alpha=0.9)
                    lastIndex = max([sample_index-1,0])
                    currentIndex = min([sample_index,len(self.yLearn)-1])
                    self.lastLearningDate = ax.scatter(self.xLearn[lastIndex][0], 
                                                       self.xLearn[lastIndex][1],
                                                       self.yLearn[lastIndex])
                    self.currentLearningDate = ax.scatter(self.xLearn[currentIndex][0],
                                                          self.xLearn[currentIndex][1],
                                                          self.yLearn[currentIndex])
                    
                    plt.draw()
                self.sfreq.on_changed(update2D)
                plt.show()
            else:
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                self.surf = ax.plot_surface(self.x_dim, 
                                        self.y_dim, 
                                        self.approximation[-1], 
                                        rstride=1, 
                                        cstride=1, 
                                        cmap=cm.coolwarm,
                                        linewidth=0, antialiased=False)
                self.lastLearningDate = ax.scatter(self.xLearn[-1][0], 
                                                   self.xLearn[-1][1], 
                                                   self.yLearn[-1])
                self.currentLearningDate = ax.scatter(self.xLearn[-1][0], 
                                                      self.xLearn[-1][1], 
                                                      self.yLearn[-1])
                   
                axcolor = 'white'
                self.axfreq = plt.axes([0.1, 0.01, 0.85, 0.03], axisbg=axcolor)
                self.sfreq = Slider(self.axfreq, '#Sample', 0, len(self.yLearn), valinit=len(self.yLearn))
                def update2D(val):
                    sample_index = int(round(self.sfreq.val))
                    self.surf.remove()
                    self.lastLearningDate.remove()
                    self.currentLearningDate.remove()
                    self.surf = ax.plot_surface(self.x_dim, 
                                        self.y_dim, 
                                        self.approximation[sample_index], 
                                        rstride=1, 
                                        cstride=1, 
                                        cmap=cm.coolwarm,
                                        linewidth=0, antialiased=False)
                    lastIndex = max([sample_index-1,0])
                    currentIndex = min([sample_index,len(self.yLearn)-1])
                    self.lastLearningDate = ax.scatter(self.xLearn[lastIndex][0], 
                                                       self.xLearn[lastIndex][1],
                                                       self.yLearn[lastIndex])
                    self.currentLearningDate = ax.scatter(self.xLearn[currentIndex][0],
                                                          self.xLearn[currentIndex][1],
                                                          self.yLearn[currentIndex])
                    plt.draw()
                self.sfreq.on_changed(update2D)
                plt.show()
        else:
            pass
        
class PlotLayerLearningHistory(BasicProcessingModule):
    """Save & plot the learning history of an layered linear simplicial B-spline learning module
    
    """
    def __init__(self,foot):
        BasicProcessingModule.__init__(self,foot)
#        self.samplingDomain = s
#        self.samplingSize = foot["samplingSize"]
        if np.size(self.samplingDomain) == 2:
            self.inputDim = 1
            self.samplingPoints = np.linspace(*self.samplingDomain,num=self.samplingSize)
            self.dimSize = self.samplingSize
        elif np.size(self.samplingDomain) == 4:
            self.inputDim = 2
            dimSize = int(np.floor(np.sqrt(self.samplingSize)))
            x = np.linspace(*self.samplingDomain[0],num=dimSize)
            y = np.linspace(*self.samplingDomain[1],num=dimSize)
            xv,yv = np.meshgrid(x,y)
            self.samplingSize = dimSize**2
            self.dimSize = dimSize
            self.x_dim = xv
            self.y_dim = yv

        self.reset()
        
    def reset(self):
        self.approximation = []
        self.xLearn = []
        self.yLearn = []
        self.layerVal = []
        self.layerPos = []
        self.layerVar = []
            
    def prepare(self,antecessor):
        observed_learning_system = antecessor["ILS"]
        if self.inputDim == 1:
            yPred = np.zeros(self.samplingSize)
            for i in np.arange(self.samplingSize):
                yPred[i] = observed_learning_system.evaluate(self.samplingPoints[i])
        elif self.inputDim == 2:
            yPred = np.zeros(np.shape(self.x_dim))
            for i in xrange(self.dimSize):
                for j in xrange(self.dimSize):
                    yPred[i,j] = observed_learning_system.evaluate(np.array([self.x_dim[i,j],self.y_dim[i,j]]))
        else:
            pass            
        self.approximation.append(yPred)
        
    def __call__(self, index=0, **argIn):
        observed_learning_system = self.inputArguments["ILS"]
        layersValue = []
        layersPosition = []
        layersVar = []
        for layer in observed_learning_system.layer:
            layersValue.append(layer.approximator.alpha.toList())
            idx = layer.approximator.alpha.toList('multiDimKey')
            pos = np.zeros(np.shape(idx))
            for i in xrange(len(pos)):
                pos[i] = layer.approximator.inputs.idx2pos(idx[i])
            layersPosition.append(pos)
            layersVar.append(layer.approximator.alpha.toList("std"))
        self.layerVal.append(layersValue)
        self.layerPos.append(layersPosition)
        self.layerVar.append(layersVar)
            
        if self.inputDim == 1:
            yPred = np.zeros(self.samplingSize)
            for i in np.arange(self.samplingSize):
                yPred[i] = observed_learning_system.evaluate(self.samplingPoints[i])
        elif self.inputDim == 2:
            yPred = np.zeros(np.shape(self.x_dim))
            for i in xrange(self.dimSize):
                for j in xrange(self.dimSize):
                    yPred[i,j] = observed_learning_system.evaluate(np.array([self.x_dim[i,j],self.y_dim[i,j]]))
        else:
            pass
            
        self.approximation.append(np.array(yPred))
        self.xLearn.append(np.array(argIn["xLearn"]))
        self.yLearn.append(np.array(argIn["yLearn"]))
        
    def end(self):
        maxNrLayer = len(self.layerVal[-1])
        if self.inputDim == 1:
            groundTruthModule = self.inputArguments["GroundTruth"]
            key = groundTruthModule.foot["input"].keys()[0]
            plt.figure()
            self.axesMain = plt.axes([0.03, 0.07, 0.45, 0.8])
            yGT = np.zeros(self.samplingSize)
            for i in np.arange(self.samplingSize):
                yGT[i] = groundTruthModule(**{key:self.samplingPoints[i], "index":len(self.yLearn)})            
            ymin = np.min(yGT)
            ymax = np.max(yGT)
            y_range = ymax-ymin
            ymin -= y_range/10
            ymax += y_range/10
            self.axesMain.set_ylim([ymin,ymax])
            self.layerAxes = []
            layer_axes_height = 0.9 / maxNrLayer
            
            spacing = 0.1 / (maxNrLayer+1)
            offset = 1 - layer_axes_height - spacing
            for i in xrange(maxNrLayer):
                self.layerAxes.append(plt.axes([0.53, offset, 0.45, layer_axes_height]))
                self.layerAxes[i].set_ylim([ymin,ymax])
                offset -= layer_axes_height + spacing
            
            self.layerLines = []
            self.layerVarLinesUpper = []
            self.layerVarLinesLower = []
            for i in xrange(maxNrLayer):
                tmpL, = self.layerAxes[i].plot(self.layerPos[-1][i], self.layerVal[-1][i])
                a = np.array(self.layerVal[-1][i]).flatten() + np.array(self.layerVar[-1][i]).flatten()
                tmpLUp = self.layerAxes[i].plot(self.layerPos[-1][i], a)
                a = np.array(self.layerVal[-1][i]).flatten() - np.array(self.layerVar[-1][i]).flatten()
                tmpLLow = self.layerAxes[i].plot(self.layerPos[-1][i], a)
                self.layerLines.append(tmpL)
                self.layerVarLinesUpper.append(tmpLUp[0])
                self.layerVarLinesLower.append(tmpLLow[0])
            
            self.L1, = self.axesMain.plot(self.samplingPoints, self.approximation[-1])
            self.L2, = self.axesMain.plot(self.samplingPoints, self.approximation[-2], '--')
            self.L3, = self.axesMain.plot(self.xLearn[-1], self.yLearn[-1], 'or')
            self.L4, = self.axesMain.plot(self.samplingPoints, yGT, '-.')
            
            axcolor = 'white'
            self.axfreq = plt.axes([0.03, 0.01, 0.45, 0.03], axisbg=axcolor)
            self.sfreq = Slider(self.axfreq, '#Sample', 0, len(self.yLearn)-1, valinit=len(self.yLearn)-1)
#            print len(self.yLearn)
#            print len(self.approximation)
            def update1D(val):
                sample_index = int(np.floor(self.sfreq.val))
                self.L1.set_ydata(self.approximation[sample_index+1])
                self.L2.set_ydata(self.approximation[sample_index])
                self.L3.set_xdata(self.xLearn[sample_index])
                self.L3.set_ydata(self.yLearn[sample_index])
                yGT = np.zeros(self.samplingSize)
                for i in np.arange(self.samplingSize):
                    yGT[i] = groundTruthModule(**{key:self.samplingPoints[i],"index":sample_index})
                self.L4.set_ydata(yGT)
                for i in range(len(self.layerPos[sample_index])):
                    self.layerLines[i].set_xdata(self.layerPos[sample_index][i])
                    self.layerLines[i].set_ydata(self.layerVal[sample_index][i])
                    
                    a = np.array(self.layerVal[sample_index][i]).flatten() + np.array(self.layerVar[sample_index][i]).flatten()
                    self.layerVarLinesUpper[i].set_xdata(self.layerPos[sample_index][i])
                    self.layerVarLinesUpper[i].set_ydata(a)
                    
                    a = np.array(self.layerVal[sample_index][i]).flatten() - np.array(self.layerVar[sample_index][i]).flatten()
                    self.layerVarLinesLower[i].set_xdata(self.layerPos[sample_index][i])
                    self.layerVarLinesLower[i].set_ydata(a)
                for i in range(len(self.layerPos[sample_index]),maxNrLayer):
                    self.layerLines[i].set_xdata([])
                    self.layerLines[i].set_ydata([])
                    
                    self.layerVarLinesUpper[i].set_xdata([])
                    self.layerVarLinesUpper[i].set_ydata([])
                    
                    self.layerVarLinesLower[i].set_xdata([])
                    self.layerVarLinesLower[i].set_ydata([])

#            print "Before on changed"
            self.sfreq.on_changed(update1D)
#            print "After on changed"
            plt.show()
        elif self.inputDim == 2:
            pass
        else:
            pass