# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:53:46 2016

@author: joschnei
"""

from __future__ import division

import numpy as np
from CPA import CPA
import itertools
import matplotlib.pyplot as plt

class CSA:
    def __init__(self, grade, func, xmin, xmax, intGrade):
        self.grade = grade
        self.func = func
        self.dim = len(xmin)
        self.xmin = xmin
        self.xmax = xmax
        self.intGrade = intGrade
        borders = []
        for i in range(self.dim):
            b = np.linspace(xmin[i],xmax[i],grade+1)
            tmp = []
            for j in range(len(b)-1):
                tmp.append([b[j], b[j+1]])
            borders.append(tmp)
        tiles = list(itertools.product(*borders))
        mins = []
        maxs = []
        for i in range(len(tiles)):
            mins.append([el[0] for el in tiles[i]])
            maxs.append([el[1] for el in tiles[i]])
            
        self.cpas = []  
        for i in range(len(mins)):
            self.cpas.append(CPA(intGrade, func, mins[i], maxs[i]))
        return

    def __getActiveCPA(self, x):
        for idx, cpa in enumerate(self.cpas):
            inner = (np.all(np.array(x)>=np.array(cpa.xmin)) and \
                     np.all(np.array(x)<=np.array(cpa.xmax)))
            if inner:
                return idx, cpa          

    def getActiveCPA(self, x):
        for idx, cpa in enumerate(self.cpas):
            inner = (np.all(np.array(x)>=np.array(cpa.xmin)) and \
                     np.all(np.array(x)<=np.array(cpa.xmax)))
            if inner:
                return idx, cpa

    def learn(self, x, y, l):
        idx, cpa = self.__getActiveCPA(x)
#        if(idx > 0):
#            cpaL = csa.cpas[idx-1]
#            cpa.alpha[0] = cpaL.alpha[-1]
#        if(idx < grade):
#            cpaH = csa.cpas[idx+1]
#        if(idx == 0):
        cpa.learn(x,y,l)
        
    
    def getPoints(self, x):
        result = []
        for el in x:
            idx, cpa = self.__getActiveCPA(el)
            result.append(cpa.getPoint(el))
        return result

    def toLearn(self, x):
        return self.func(x)   

def func(x):
    return [1]*len(x)#np.sin(x)

if __name__ == "__main__":
    xmin = [-10]
    xmax = [10]
    csa = CSA(1,0,[-10],[10],30)
    xint = np.subtract(xmax, xmin)
    numTrain = 10000
    res = 200
    x = np.linspace(-10,10,res)
    xtrain = np.random.rand(numTrain)*xint[0]-xint[0]/2
    ytrain = func(xtrain)

    train = zip(xtrain, ytrain)
    for idx, el in enumerate(train):
        print idx+1, '/', numTrain
        csa.learn(el[0], el[1], 0.1)
    y = csa.getPoints(x)
    plt.plot(x,y)
