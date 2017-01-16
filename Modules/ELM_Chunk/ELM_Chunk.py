# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 17:49:03 2016

@author: Henning
"""
from __future__ import division
from Modules.IncrementalLearning.IncrementalLearningSystem import IncrementalLearningSystem 
import numpy as np
import time
from ActivationFunctions import Sigmoid

class ELM_Chunk ( IncrementalLearningSystem ):
    
    def init ( self , foot ): 
        self.foot= foot
        IncrementalLearningSystem.init( self , foot )
        
    def prepare ( self , antecessor ): 
        self.nrIn = len(antecessor["xLearn"].output)
        self.nrHidden = 20
        #self.wIn = np.random.randn(self.nrHidden,self.nrIn)
        self.wIn = np.ones((self.nrHidden,self.nrIn))
        self.wOut = np.random.randn(self.nrHidden,1)
        self.nodes = np.empty(self.nrHidden,dtype=object)
        for ni,_ in enumerate(self.nodes):
            #zufaelliges SIgmoid erstellen
            sig = Sigmoid()
            #sig = self.TanH()
            self.nodes[ni] = sig
        self.H = None
        #from paper: nr of initial trainingdata not less than nr of hidden nodes
        self.nrInitT = self.nrHidden*2
        #self.nrInitT = 70080
        self.initT = []
        self.counter = 0
        print("PREPARE ENDE")
        
    def normalize(self,x):
        return x/3.14
            
            
    def evaluate ( self , x ): 
        x = self.normalize(x)
        
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
        x = self.normalize(x)
        in0 = np.dot(self.wIn,np.transpose(x))
        dH = np.empty((1,self.nrHidden),dtype=np.float64)
        for i in range(self.nrHidden):
            dH[0][i] = self.nodes[i].evaluate(in0[i])
       
        if self.H == None:
            self.H = np.copy(dH)
            sizeH = 1
        else:
            #TODO: self.H muss gar nicht erweitert werden, wenn schon in online phase
            if self.H.shape[0] <= self.nrInitT:            
                self.H = np.vstack((self.H,dH))
            sizeH = self.H.shape[0]
        #print("H.shape: ",np.shape(self.H))
        if sizeH <= self.nrInitT:
            self.initT.append(y)
            
         #falls erst nrInitT trainingsdaten:
        if sizeH == self.nrInitT:
#            print("learn on init")
#            print(dH)
#            print(self.H)
            initT = np.array(self.initT,dtype=np.float64).reshape((self.nrInitT,1))
            #print("vor: ",self.wOut)
            #wOut nach klassischer Batch Berechnung
            self.wOut = np.dot(np.linalg.pinv(self.H), initT)

            #print("nach: ",self.wOut)
            toInv = np.dot(np.transpose(self.H), self.H)
            #self.P = np.linalg.pinv(toInv+np.identity(len(toInv)))
            self.P = self.invSVD(np.identity(len(toInv))+toInv) #np.identity(len(toInv))+
            print("P nach init: ",self.P)
            print("H nach init: ",self.H)
            
            #wOut nach online in batch phase:
            #self.wOut = np.dot(np.dot(self.P,np.transpose(self.H)),initT)
            
            print("Iself.wOut: ",self.wOut)            
        #sonst: (one by one version) 
        elif sizeH >self.nrInitT:
            #print("learn online")
            dHtrans = np.transpose(dH)
            Pt0 = np.dot(self.P,dHtrans) 
            Pt1 = np.dot(np.dot(dH,self.P),dHtrans)
            #Pt1 = np.linalg.pinv(np.identity(len(Pt1)) + Pt1)
            Pt1 = self.invSVD(np.identity(len(Pt1)) + Pt1)
            
            self.P = self.P - np.dot(np.dot(np.dot(Pt0,Pt1),dH),self.P)
            #self.P = self.P + np.identity(len(self.P))
            wOut0 = np.dot(self.P,dHtrans)
            wOut1 = (y - np.dot(dH,self.wOut))
            #print("wOut vor update :",self.wOut)
            self.wOut = self.wOut + np.dot(wOut0,wOut1)
        self.counter+=1
        
    def invSVD(self,toInv):
    
        u,s,v=np.linalg.svd(toInv)
        inverse =np.dot(v,np.dot(np.linalg.inv(np.diag(s)),np.transpose(u)))
        return inverse
    
