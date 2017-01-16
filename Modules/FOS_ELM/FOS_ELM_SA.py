# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 17:49:03 2016

@author: Henning
"""
from __future__ import division
from Modules.IncrementalLearning.IncrementalLearningSystem import IncrementalLearningSystem 
import numpy as np
from numpy.linalg import pinv
#from numpy.linalg import inv as pinv
#from ActivationFunctions import invSVD as pinv
from ActivationFunctions import TanH
rcond = 1e-10

class FOS_ELM_SA( IncrementalLearningSystem ):
    def init ( self , foot ): 
        self.foot= foot
        IncrementalLearningSystem.init( self , foot )

    def prepare ( self , antecessor ): 

        self.nrIn = len(antecessor["xLearn"].output) + 1
     
        self.nrHidden = 30
        self.window = 200
        
        self.wIn = np.random.randn(self.nrHidden,self.nrIn)
        #self.wIn = np.ones((self.nrHidden,self.nrIn))
        self.wOut = np.random.randn(self.nrHidden,1)
        self.nodes = np.empty(self.nrHidden,dtype=object)
        for i in range(self.nrHidden):
            #zufaelliges SIgmoid erstellen
            self.nodes[i] = TanH()
        self.H = None
        #from paper: nr of initial trainingdata not less than nr of hidden nodes
        self.nrInitT = self.nrHidden *2
        self.initT = []
        self.counter = 0
    def normalize(self,x):
        #x = x/6.28
        xn = np.ones(self.nrIn)
        xn[0:-1] = x
        return xn
            
            
    def evaluate ( self , x ): 
        #jeder input ist mit jedem hidden node verknuepft
        #algebraische Summe als Eingang fuer Aktivierung
        x = self.normalize(x)
        in0 = np.dot(self.wIn,np.transpose(x))
        Ha = np.empty((1,self.nrHidden),dtype=np.float64)
        for i in range(self.nrHidden):
            Ha[0][i] = self.nodes[i].evaluate(in0[i])
        out = np.dot(Ha,self.wOut)
        return out
        
    def learn ( self , x , y ):
        x = self.normalize(x)
        in0 = np.dot(self.wIn,np.transpose(x))
        dH = np.empty((1,self.nrHidden),dtype=np.float64)
        for i in range(self.nrHidden):
            dH[0][i] = self.nodes[i].evaluate(in0[i])
       
        if self.H == None:
            self.H = np.copy(dH)
            sizeH = 1
        else:
            self.H = np.vstack((self.H,dH))
            sizeH = self.H.shape[0]
        self.initT.append(y)
            
         #falls erst nrInitT trainingsdaten:
        if sizeH == self.nrInitT:
            print("init lernen")
            initTrain = np.array(self.initT,dtype=np.float64).reshape((self.nrInitT,1))
            toInv = np.dot(np.transpose(self.H), self.H)
            self.P = pinv(toInv,rcond=rcond)
            #wOut nach online in batch phase:
            self.wOut = np.dot(np.dot(self.P,np.transpose(self.H)),initTrain)

        elif sizeH > self.nrInitT and sizeH < self.window:
            dHtrans = np.transpose(dH)
            Pt0 = np.dot(self.P,dHtrans) 
            Pt1 = np.dot(np.dot(dH,self.P),dHtrans)
            Pt1 = pinv(np.identity(len(Pt1)) + Pt1,rcond=rcond)
            
            self.P = self.P - np.dot(np.dot(np.dot(Pt0,Pt1),dH),self.P)
            wOut0 = np.dot(self.P,dHtrans)
            wOut1 = (y - np.dot(dH,self.wOut))
            self.wOut = self.wOut + np.dot(wOut0,wOut1)
            
        #sonst: (online forgetting phase) 
        elif sizeH > self.nrInitT and sizeH >= self.window:
            #print("lernen mit vergessen")
            Hf = np.transpose(np.vstack((-1*self.H[0],dH)))
            Hn = np.vstack((self.H[0],dH))
            
            P0 = np.dot(self.P, Hf)
            P1 = np.dot(np.dot(Hn,self.P), Hf)
            P2 = pinv(np.identity(len(P1)) + P1,rcond=rcond)
            #print(np.linalg.det(np.identity(len(P1)) + P1))
            P3 = np.dot(P0,P2)
            self.P = self.P - np.dot(np.dot(P3,Hn),self.P)
            
            #self.P = pinv(pinv(self.P) + np.dot(Hf,Hn))            
            
            initTA = np.vstack((self.initT[0],self.initT[-1]))         
            
            wOut0 = np.dot(self.P,Hf)
            wOut1 = initTA - np.dot(Hn,self.wOut)

            self.wOut = self.wOut + np.dot(wOut0,wOut1)
            
            self.H = np.delete(self.H,0,0)
            self.initT = self.initT[1::]

        self.counter+=1
    

        
    def reset(self):
        self.nodes = np.empty(self.nrHidden,dtype=object)
        self.wIn = np.zeros((self.nrHidden,self.nrIn))
        self.wOut = np.zeros((self.nrHidden,1))
        self.nrHidden = 0
        self.window = 0
        self.nrIn = 0        
        self.H = None
        self.nrInitT = 0
        self.initT = []
        self.counter = 0
    def __str__(self):
        return str(self.wIn)


    
