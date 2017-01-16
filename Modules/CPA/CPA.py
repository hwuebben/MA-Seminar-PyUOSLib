# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 14:00:00 2016
"""

## @package CPA
#  Chebyshev Polynomial Approximator
#  This class contains the algorithms for a Chebyshev Polynomial Approximator
#  The transformation function phi is split up into phi_learn and phi_eval
#  The learning process works on a  simple local GLT approximator.
#  During evaluation the nodes of the GLT approximators are treated as sampling
#  points for a polynomial approximator. Due to the positioning of the nodes
#  corresponding to the zero points of the Chebyshev differential equation the
#  condition number of the polynomial fit is 1 for the 1-dimensional case and
#  for higher dimensionalities much smaller than comparable equidistantly placed
#  approximator nodes.
#
#  @author: Jonas Schneider


from __future__ import division
import numpy as np
from Chebyshev import Chebyshev

## Class representing an n-dimensional Chebyshev-based approximator.
#  Complete approximator with first-order Passive-Agressive-Learning and
#  Second-Order RLS-Learning.
class CPA:
    ## The constructor.
    #  @param self The object pointer
    #  @param grade Grade of the polynomial (same for all dimensions)
    #  @param func Target function (function pointer)
    #  @param xmin Array consisting of minima of dimensions [xmin_0 xmin_1 ...]
    #  @param xmin Array consisting of maxima of dimensions [xmax_0 xmax_1 ...]
    def __init__(self, grade, func, xmin, xmax):
        # The grade of the polynomial per dimension
        # (same grade for each dimension)
        if len(xmin) != len(xmax):
            print 'xmin and xmax size mismatch. Terminating.'
            exit()
        else:
            self.dim = len(xmin)
        self.grade = grade
        # Limits of the intervals of the input space per dimension
        # xmin: [xmin_0 xmin_1 ...]
        # xmax: [xmax_0 xmax_1 ...]
        self.xmin = xmin
        self.xmax = xmax
        # Total number of parameters (equals total number of grid points)
        self.numParam = (self.grade+1)**self.dim
        # Initialization of the parameter vector
        self.alpha = [0]*self.numParam
        # Initialization of local density
#        self.locDensity = [0]*self.numParam
#        # Additive local variance
#        self.locVariance = [0]*self.numParam
#        self.locVarianceAbs = [0]*self.numParam
#        # Further local measures
#        self.locIgnorance = [0]*self.numParam
#        self.locConflict = [0]*self.numParam
#        self.etaI = 0.001
#        self.etaC = 0.001
        # Set the target function
        self.func = func
        # Initialization of the ground truth error
        self.errorGT = 0
        # Calculates the position of the chebyshev points for the grade of the
        # polynomials
        self.cheby = Chebyshev(grade)
        # TODO: Following line needs to be adapted for >1D
#        self.chebyPoints = self.__getGlobalX(self.cheby.points)

    ## Reset the knowledge. Set parameter vector to zero
    #  @param self The object pointer
    def reset(self):
        # Re-Initialisation of the parameter vector
        self.alpha = [0]*self.numParam
        # Initialization of local density
#        self.locDensity = [0]*self.numParam
#        # Additive local variance
#        self.locVariance = [0]*self.numParam
#        self.locVarianceAbs = [0]*self.numParam
#        # Further local measures
#        self.locIgnorance = [0]*self.numParam
#        self.locConflict = [0]*self.numParam
#        self.etaI = 0.001
#        self.etaC = 0.001
        # Initilization of the ground truth error
        self.errorGT = 0
        
    ## Transform a global point into coordinate on unit hypersphere
    #  @param self The object pointer
    #  @param x The coordinate in global coordinates that is to be transformed
    #  @return The transformed local coordinate
    def __getX(self, x):
        xint = np.subtract(self.xmax,self.xmin)
        xtmp = np.subtract(x,self.xmin)
        return np.true_divide(xtmp,xint) * 2 - 1
    ## Transforms a local coordinate into coordinate in input space
    #  @param self The object pointer
    #  @param x The coordinate in local coordinates that is to be transformed
    #  @return The transformed global coordinate
    def __getGlobalX(self, x):
        xint = np.subtract(self.xmax,self.xmin)
        return np.add(x,1)*xint/2+self.xmin
    ## Learn via Passive-Agressive in hypersphere
    #  Private method because coordinates are coordinates in hypersphere
    #  @param self The object pointer
    #  @param x Coordinate of training datum in hypersphere
    #  @param y Desired output value
    #  @param l Learning rate (0<=l<=1)
    def __learn(self, x, y, l):
        (llcoord, phix) = self.__transform(x)
#        self.locDensity[llcoord] += phix[llcoord]
        dy = y - self.__evalGLT(x)
#        self.locVariance[llcoord] += dy
#        self.locVarianceAbs[llcoord] += np.absolute(dy)
        self.alpha = np.add(self.alpha, np.multiply(phix, dy)*l)
#        self.locIgnorance[llcoord] = (self.etaI*self.locDensity[llcoord]) / (1.0+self.etaI*self.locDensity[llcoord])
#        self.locConflict[llcoord] = 1.0 / (1+self.etaC*(1.0/self.locDensity[llcoord])*self.locVariance[llcoord])

    ## Learn via Passive-Agressive
    #  @param self The object pointer
    #  @param x Coordinate of training datum in input space
    #  @param y Desired output value
    #  @param l Learning rate (0<=l<=1)        
    def learn(self, x, y, l):
        newx = self.__getX(x)
        self.__learn(newx, y, l)

#    def __learnLocalRLS(self, x, y):
#        (llcoord, phix) = self.__transform(x)
#        self.locDensity[llcoord] += phix[llcoord]
#        dy = y - self.__evalGLT(x)
#        self.locVariance[llcoord] += dy
#        self.locVarianceAbs[llcoord] += np.absolute(dy)
#        l = 1.0/(1.0+self.locConflict[llcoord])
#        self.alpha = np.add(self.alpha, np.multiply(phix, dy)*l)
#        self.locIgnorance[llcoord] = (self.etaI*self.locDensity[llcoord]) / (1.0+self.etaI*self.locDensity[llcoord])
#        self.locConflict[llcoord] = 1.0 / (1+self.etaC*(1.0/self.locDensity[llcoord])*self.locVarianceAbs[llcoord])
#
#    def learnLocalRLS(self, x, y):
#        newx = self.__getX(x)
#        self.__learnLocalRLS(newx, y)

    ## Transformation function phi_learn
    #  Private method because coordinates are coordinates in hypersphere
    #  CAUTION: Do NOT call this function with coordinates in input space!
    #  @param self The object pointer
    #  @param x Coordinate in hypersphere
    #  @return phi_learn(x)
    def __transform(self, x):
        coordPerDim = [0]*self.dim
        result = [0]*self.numParam
        actDim = [0]*self.dim
        dist = np.zeros((self.dim,2))
        for d in range(self.dim):
            if x[d] < self.cheby.points[0]:
                coordPerDim[d] = 0
                actDim[d] = -1
                dist[d][0] = np.abs(x[d]-self.cheby.points[0])
                dist[d][1] = np.abs(x[d]-self.cheby.points[0])
            elif x[d] > self.cheby.points[-1]:
                coordPerDim[d] = self.grade
                actDim[d] = -2
                dist[d][0] = np.abs(x[d]-self.cheby.points[-1])
                dist[d][1] = np.abs(x[d]-self.cheby.points[-1])
            else:
                for i in range(self.grade):
                    if(x[d] >= self.cheby.points[i] and x[d] <= self.cheby.points[i+1]):
                        coordPerDim[d] = i
                        dist[d][0] = np.abs(x[d]-self.cheby.points[i])
                        dist[d][1] = np.abs(x[d]-self.cheby.points[i+1])
                        if(dist[d][0] >= dist[d][1]):
                            actDim[d] = 1
                        else:
                            actDim[d] = 0
                                   
        tmp = 0
        tmpHalf = 0
        for d in range(self.dim):
            tmp += dist[d][actDim[d]]**2
            if(actDim[d] < 0):
                tmpHalf += self.cheby.chebyDistHalf[0]
            else:
                tmpHalf += self.cheby.chebyDistHalf[coordPerDim[d]+1]
        tmp = np.sqrt(tmp)
        tmpHalf = np.sqrt(tmpHalf)
        act = 1 - (tmp / tmpHalf)
        actDim = np.maximum(actDim,0)
        llcoord = 0
        for d in range(self.dim):
            llcoord += (coordPerDim[d]+actDim[d])*(self.grade+1)**d
        result[llcoord] = act
        return (llcoord, result)

    ## Evaluation of the GLT
    #  returns phi_learn(x)*alpha
    #  Private method because coordinates are coordinates in hypersphere
    #  @param self The object pointer
    #  @param x Coordinate in hypersphere
    #  @return phi_learn(x)*alpha
    def __evalGLT(self, x):
        phi = self.__transform(x)[1]
        return np.dot(phi,self.alpha)
        
    ## Evaluation of the Polynomial via Lagrange-Basis
    #  returns phi_eval(x,nodes,alpha)
    #  Private method because coordinates are coordinates in hypersphere
    #  @param self The object pointer
    #  @param x Coordinate in hypersphere
    #  @return phi_eval(x,nodes,alpha)
    def __evalPol(self, x):
        
        # faster algorithm O(n^2) which directly calculates the output without
        # the polynom coefficients
        
        if self.dim == 1:
            l = [1]*len(self.cheby.points)
            for i in range(len(self.cheby.points)):
                for j in range(len(self.cheby.points)):
                    if i == j:
                        continue
                    else:
                        l[i] *= (x-self.cheby.points[j])/(self.cheby.points[i]-self.cheby.points[j])
            return np.dot(self.alpha, l)
         
        else:
            l = np.ones([self.dim,len(self.cheby.points)])   
            for dim in range(self.dim):
                for i in range(len(self.cheby.points)):
                    for j in range(len(self.cheby.points)):
                        if i == j:
                            continue
                        else:   
                            l[dim,i] *= (x[dim]-self.cheby.points[j])/(self.cheby.points[i]-self.cheby.points[j])
            mat = np.outer(l[0,:],l[1,:])
            for i in range(1, dim):
                mat = np.outer(mat,l[i+1,:])
            return np.dot(mat.flatten(),self.alpha)

    ## Returns the approximation of every point in an array
    #  @param self The object pointer
    #  @param x Array of query points in input space
    #  @return Array of approximation results
    def getPoints(self, x):
        result = []
        newx = self.__getX(x)
        for i in newx:
            result.append(self.__evalPol(i))
        return result
        
    ## Returns the approximation of one point
    #  @param self The object pointer
    #  @param x query point in input space
    #  @return approximation result
    def getPoint(self, x):
        newx = self.__getX(x)
        return self.__evalPol(newx)
    
    ## Returns the value that is to be learned for an input value, i.e.
    #  the ground truth value
    #  @param self The object pointer
    #  @param The query point in input space
    #  @return The ground truth value for the query point
    def toLearn(self, x):
        return self.func(x)
    
    ## Returns the values that are to be learned for an array of input values,
    #  i.e. the ground truth value
    #  @param self The object pointer
    #  @param The query points in input space
    #  @return The ground truth values for the query points
    def getToLearnPoints(self, x):
        result = []
        for i in x:
            result.append(self.toLearn(i))
        return result
    
    ## Calculates the ground truth error for the current approximation on
    #  the array of query points in input space
    #  @param self The object pointer
    #  @param x Single or array of input points in input space (typically dense and equidistant points)
    #  @return Squared mean ground truth error
    def calcError(self, x):
        yGT = self.getToLearnPoints(x)
        yL = self.getPoints(x)
        self.errorGT = np.sqrt(np.sum(np.power(np.subtract(yGT,yL),2))/len(x))
        return self.errorGT

    ## Calculates and returns the matrix describing the linear equation system
    #  for the calculation of the polynomial parameters (standard polynomial)
    #  @param self The object pointer
    #  @return The linear equation system matrix     
    def getLESMatrix(self):
        #TODO: CHECK FOR CORRECT FUNCTION IF >1D
        result = []
        for pt in self.cheby.points:
            tmp = []
            for i in range(self.grade+1):
                tmp.append(pt**i)
            result.append(tmp)
        tmp = result
        for i in range(1, self.dim):
            result = np.outer(result, tmp)
        return result
    ## Returns the condition number of the underlying linear equation system
    #  @param self The object pointer
    #  @return The condition number
    def getCondNr(self):
        return np.linalg.cond(self.getLESMatrix())

    ## Returns the polynomial coefficients (standard polynomial)
    #  @param self The object pointer
    #  @return Array of polynomial coefficients
    def getStandardPolynomialCoefficients(self):
        result = np.linalg.inv(self.getLESMatrix())
        result = np.dot(result, self.alpha)
        return result
