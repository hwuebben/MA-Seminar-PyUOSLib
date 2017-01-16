# -*- coding: utf-8 -*-
"""
Created on Sat Jan 07 17:53:48 2017

@author: Henning
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 17:49:03 2016

@author: Henning
"""
from __future__ import division
from Modules.IncrementalLearning.IncrementalLearningSystem import IncrementalLearningSystem 
import numpy as np
import time

class ELM ( IncrementalLearningSystem ):
    
    def init ( self , foot ): 
        IncrementalLearningSystem.init( self , foot )
        
    def prepare ( self , antecessor ): 
        
        self.nrIn = len(antecessor["xLearn"].output)
        self.nrHidden = 7
        self.wIn = np.random.randn(self.nrHidden,self.nrIn)
        self.wOut = np.random.randn(self.nrHidden,1)
        self.nodes = np.empty(self.nrHidden,dtype=object)
        for ni,_ in enumerate(self.nodes):
            #zufaelliges SIgmoid erstellen
            sig = self.Sigmoid(np.random.normal(),np.random.normal())
            #sig = self.TanH()
            self.nodes[ni] = sig
        self.H = None
        #from paper: nr of initial trainingdata not less than nr of hidden nodes
        self.nrInitT = self.nrHidden
        self.initT = []
        self.counter = 0
        print("PREPARE ENDE")
            
            
    def evaluate ( self , x ): 
        #print("start evaluating")
        #jeder input ist mit jedem hidden node verknuepft
        #algebraische Summe als Eingang fuer Aktivierung
        in0 = np.dot(self.wIn,np.transpose(x))
        Ha = np.empty((1,self.nrHidden),dtype=np.float64)
        for i in range(self.nrHidden):
            Ha[0][i] = self.nodes[i].evaluate(in0[i])
        out = np.dot(Ha,self.wOut)
        #print("returned: ",out)
        return out
        
    def learn ( self , x , y ):

        in0 = np.dot(self.wIn,np.transpose(x))
        dH = np.empty((1,self.nrHidden),dtype=np.float64)
        for i in range(self.nrHidden):
            dH[0][i] = self.nodes[i].evaluate(in0[i])
       
        if self.H == None:
            self.H = np.copy(dH)
            sizeH = 1
        else:
            if self.H.shape[0] <= self.nrInitT:            
                self.H = np.vstack((self.H,dH))
            sizeH = self.H.shape[0]
        if sizeH <= self.nrInitT:
            self.initT.append(y)
            
         #falls erst nrInitT trainingsdaten:
        if sizeH == self.nrInitT:
            
            initT = np.array(self.initT,dtype=np.float64).reshape((self.nrInitT,1))
            toInv = np.dot(np.transpose(self.H), self.H)
            self.P = np.linalg.inv(toInv)
            self.wOut = np.dot(np.dot(self.P,np.transpose(self.H)),initT)

        #sonst: (one by one version) 
        elif sizeH >self.nrInitT:
            #print("learn online")
            dHtrans = np.transpose(dH)      
            
            Pt0 = np.dot(np.dot(np.dot(self.P,dHtrans),dH),self.P)
            Pt1 = np.float64(1.0 + np.dot(np.dot(dH,self.P),dHtrans))
            self.P = self.P - np.true_divide(Pt0, Pt1)
            
            wOut0 = np.dot(self.P,dHtrans)
            wOut1 = np.float64(y - np.dot(dH,self.wOut))
            self.wOut = self.wOut + wOut0 * wOut1

        self.counter+=1
        if self.counter == 400:
            print("wOut am Ende: ",self.wOut)
        

        
    class TanH:
        def __init__(self):
            pass        
        def evaluate(self,x):
            return np.tanh(x)
    
    class Sigmoid:
        def __init__(self, a, b):
            self.a = a
            self.b = b
        def evaluate(self,x):
            sigX = 1/( 1+np.exp( -(self.a*x+self.b) ) )
            return sigX
        def __str__(self):
            return "a: "+str(self.a)+" b: "+str(self.b)
            
if __name__ == "__main__":
    sig = ELM.Sigmoid(-0.4,1.6)
    print(sig.evaluate(3))
    
