# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 17:49:03 2016

@author: Henning
"""
from __future__ import division
from Modules.IncrementalLearning.IncrementalLearningSystem import IncrementalLearningSystem 
import numpy as np
from ActivationFunctions import Sigmoid

class ELM ( IncrementalLearningSystem ):
    
    def init ( self , foot ): 
        IncrementalLearningSystem.init( self , foot )
        #size of the hidden layer:
        self.nrHidden = getattr(foot["parameter"], "nrHidden")
        
    def prepare ( self , antecessor ): 
        
        #length of the inputs:
        self.nrIn = len(antecessor["xLearn"].output)
        #input weights are one:
        self.wIn = np.ones((self.nrHidden,self.nrIn))
        #randomly init output weights:
        self.wOut = np.random.randn(self.nrHidden,1)
        #create random sigmoids for the hidden nodes:
        self.nodes = np.empty(self.nrHidden,dtype=object)
        for ni,_ in enumerate(self.nodes):
            sig = Sigmoid()
            self.nodes[ni] = sig
        self.H = None
        #from paper: nr of initial trainingdata not less than nr of hidden nodes
        self.nrInitT = self.nrHidden
        self.initT = []
            
            
    def evaluate ( self , x ): 
        #each input is connected with each hidden node, algebraic sum activated:
        in0 = np.dot(self.wIn,np.transpose(x))
        #calculate activation of the nodes:
        Ha = np.empty((1,self.nrHidden),dtype=np.float64)
        for i in range(self.nrHidden):
            Ha[0][i] = self.nodes[i].evaluate(in0[i])
        #calculate output:
        out = np.dot(Ha,self.wOut)
        return out
        
    def learn ( self , x , y ):
        #each input is connected with each hidden node, algebraic sum activated:
        in0 = np.dot(self.wIn,np.transpose(x))
        #calculate activation of the nodes:
        dH = np.empty((1,self.nrHidden),dtype=np.float64)
        for i in range(self.nrHidden):
            dH[0][i] = self.nodes[i].evaluate(in0[i])
        #if H doesnt exist: create it
        if self.H == None:
            self.H = np.copy(dH)
            sizeH = 1
        else:
            #if still collecting data for batch learning: append dH to H
            if self.H.shape[0] <= self.nrInitT:            
                self.H = np.vstack((self.H,dH))
            sizeH = self.H.shape[0]
        #collect data for batch phase:
        if sizeH <= self.nrInitT:
            self.initT.append(y)
            
         #batch learning on the first nrInit data points:
        if sizeH == self.nrInitT:
            initT = np.array(self.initT,dtype=np.float64).reshape((self.nrInitT,1))
            #wOut with classical batch learning rule:
            self.wOut = np.dot(np.linalg.pinv(self.H), initT)
            
            #initialize P for online phase
            toInv = np.dot(np.transpose(self.H), self.H)
            self.P = np.linalg.inv(toInv)

        #else: (one by one version) 
        elif sizeH >self.nrInitT:
            
            dHtrans = np.transpose(dH)
            #update P:
            Pt0 = np.dot(np.dot(np.dot(self.P,dHtrans),dH),self.P)
            Pt1 = np.float64(1.0 + np.dot(np.dot(dH,self.P),dHtrans))
            self.P = self.P - np.true_divide(Pt0, Pt1)
            #update wOut:
            wOut0 = np.dot(self.P,dHtrans)
            wOut1 = np.float64(y - np.dot(dH,self.wOut))
            self.wOut = self.wOut + wOut0 * wOut1