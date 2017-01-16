# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 12:24:00 2016

@author: Jonas Schneider
"""

import numpy as np
import itertools

"""
This class represents a local regularization scheme based on Tikhonov
regularization. The approach is explained in the master thesis of Jonas
Schneider "Sichere globale Inter- und Extrapolation unter Nutzung lokalen
Vorwissens"
"""
class LocalRegularization:
    """
    Initializer
    @param grade Number of nodes per dimension
    @param dim Dimensionality of the input space
    """    
    def __init__(self, grade, dim):
        self.dim = dim
        self.grade = grade
        self.numParam = (self.grade)**self.dim
        self.ltEl = 3**dim
        self.mdEl = (3**dim)/2
        mask = np.zeros(3**self.dim+1)
        mask[self.mdEl] = 1
        self.regMasks = []
        for i in xrange(self.numParam):
            self.regMasks.append([mask])
        self.M = np.zeros([self.numParam+1,self.numParam+1])
        self.psi = np.zeros(self.numParam+1)
        self.psi[-1] = 1
        self.__calcNeighborOffsets()
        self.anchor = np.zeros(self.numParam)
        self.sigmaA = np.zeros(self.numParam)
        self.sigmaG = np.ones(self.numParam)
        self.alpha = np.ones(self.numParam)

    """
    Helper function to calculate the offsets for filling the one-dimensional
    indices of the matrix rows
    """
    def __calcNeighborOffsets(self):
        offsetMat = np.zeros([self.dim,3])
        for i in xrange(self.dim):
            offsetMat[i][0] = -(self.grade)**i
            offsetMat[i][1] = 0
            offsetMat[i][2] = (self.grade)**i
        self.offsets = np.zeros(3**self.dim)
        self.offsets = list(itertools.product(*offsetMat))
        self.offsets = np.sum(self.offsets,1)

    """
    Helper function that constructs one template for all nodes out of
    possibly several templates
    """
    def __constructMatrix(self):
        result = []
        for i in self.regMasks:
            # if only one template exists, use that
            if(len(i) == 1):
                result.append(i[0])
            # if more than one template exists, use the mean of all templates
            else:
                result.append(np.mean(i,axis=0))
        return result
    
    """
    Helper function to construct the matrix M and the target vector psi of the
    linear equation system
    @param omega parameter vector after the learning step
    """    
    def __constructLES(self, omega):
        masks = self.__constructMatrix()
        for i in xrange(self.numParam):
            for j in xrange(3**self.dim):
                if(i+self.offsets[j] >= 0 and i+self.offsets[j] < self.numParam):
                    # central element: blending to 1
                    if(self.offsets[j] == 0):
                        self.M[i][i+self.offsets[j]] = self.sigmaA[i]+(1-self.sigmaA[i])*\
                        ((self.sigmaG[i]*(-(masks[i][j]-1)))+\
                        (1-self.sigmaG[i])*\
                        (self.alpha[i]*(-(masks[i][j]-1))+(1-self.alpha[i])))
                    # non-central element: blending to 0
                    else:
                        self.M[i][i+self.offsets[j]] = (1-self.sigmaA[i])*\
                        (self.sigmaG[i]*(-(masks[i][j]))+\
                        (1-self.sigmaG[i])*self.alpha[i]*-(masks[i][j]))
            self.M[i][-1] = (1-self.sigmaA[i])*(self.sigmaG[i]*(-masks[i][-1]) + (1-self.sigmaG[i])*self.alpha[i]*(-masks[i][-1]))
            if(not np.any(self.M[i])):
                self.M[i][i] = 1
                self.psi[i] = omega[i]
            else:
                self.psi[i] = (self.sigmaA[i]*self.anchor[i])+(1-self.sigmaA[i])*((1-self.sigmaG[i])*(1-self.alpha[i])*omega[i])
        self.M[-1][-1] = 1        


    """
    Set a template for a given node index
    @param idx Index of the node where the template is to be set
    @param template Array containing the template
    """
    def setTemplate(self, idx, template):
        for el in template:
            if(len(el) != 3**self.dim+1):
                print 'Wrong template size. Got:', len(template),' expected:', 3**self.dim+1
                return
        self.regMasks[idx] = template

    """
    Perform a regularization step on the given paramter vector
    @param omega The parameter vector that is to be regularized
    """
    def regularize(self, omega):
        self.__constructLES(omega)
        alpha = np.linalg.solve(self.M, self.psi)
        return alpha[:-1]
    
    """
    Set an anchor value for a given node index
    @param idx Index of the node where the anchor value is to be set
    @param aleph Anchoring strength [0;1]
    @param anchor Anchoring value
    """
    def setAnchor(self, idx, aleph, anchor):
        self.anchor[idx] = anchor
        self.sigmaA[idx] = aleph
    
    """
    Set a hard anchor for a given node index
    @param idx Index of the node where the hard anchor value is to be set
    @param anchor Anchoring value
    """
    def setHardAnchor(self, idx, anchor):
        self.setAnchor(idx, 1.0, anchor)

        
if __name__ == "__main__":
    lr = LocalRegularization(4,1)
    lr.setTemplate(1,[[0,0,0.9,0]])
#    lr.setHardAnchor(0, 2)
    alpha = lr.regularize([1.6,0,-1.8,0])
    print alpha