# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 17:49:03 2016

@author: Henning
"""
from __future__ import division
from Modules.IncrementalLearning.IncrementalLearningSystem import IncrementalLearningSystem 
import numpy as np
import time

class ELM_Chunk ( IncrementalLearningSystem ):
    
    def init ( self , foot ): 
        IncrementalLearningSystem.init( self , foot )
        
    def prepare ( self , antecessor ): 
        
        self.nrIn = len(antecessor["xLearn"].output)
        self.nrHidden = 8
        self.wIn = np.random.randn(self.nrHidden,self.nrIn)
        self.wOut = np.ones(self.nrHidden,1)
        self.nodes = np.empty(self.nrHidden,dtype=object)
        for ni,_ in enumerate(self.nodes):
            #zufaelliges SIgmoid erstellen
            sig = self.Sigmoid(np.random.normal(),np.random.normal())
            self.nodes[ni] = sig
        self.H = None
        #from paper: nr of initial trainingdata not less than nr of hidden nodes
        self.nrInitT = self.nrHidden * 10
        self.initT = []
        self.counter = 0
        print("PREPARE ENDE")
            
            
    def evaluate ( self , x ): 
        #print("start evaluating")
        #jeder input ist mit jedem hidden node verknuepft
        #algebraische Summe als Eingang fuer Aktivierung
        in0 = np.dot(self.wIn,np.transpose(x))
        Ha = np.empty((1,self.nrHidden))
        for i in range(self.nrHidden):
            Ha[0][i] = self.nodes[i].evaluate(in0[i])
        out = np.dot(Ha,self.wOut)
        #print("returned: ",out)
        return out
        
    def learn ( self , x , y ):
        self.counter+=1
#        if self.counter == 8050:
#            print("P am Ende: ",self.P)
        in0 = np.dot(self.wIn,np.transpose(x))
        dH = np.empty((1,self.nrHidden))
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
            initT = np.array(self.initT).reshape((self.nrInitT,1))
            #print("vor: ",self.wOut)
            #wOut nach klassischer Batch Berechnung
            #self.wOut = np.dot(np.linalg.pinv(self.H), initT)

            #print("nach: ",self.wOut)
            toInv = np.dot(np.transpose(self.H), self.H)
            self.P = np.linalg.inv(toInv)
            print("P nach init: ",self.P)
            #wOut nach online in batch phase:
            #print("shapes: ",self.P.shape," ",self.H.shape," ",initT.shape)
            #print(np.dot(np.dot(self.P,np.transpose(self.H)),initT).shape)
            self.wOut = np.dot(np.dot(self.P,np.transpose(self.H)),initT)
            print("Iself.wOut: ",self.wOut)
            #print("initt: ",np.shape(initT))
            #print("initT: ",self.initT)
            #time.sleep(5)

        #sonst: (one by one version) 
        elif sizeH >self.nrInitT:
            #print("learn online")
            dHtrans = np.transpose(dH)
            Pt0 = np.dot(self.P,dHtrans) 
            Pt1 = np.dot(np.dot(dH,self.P),dHtrans)
            Pt1 = np.linalg.inv(np.identity(len(Pt1)) + Pt1)
            self.P = self.P - np.dot(np.dot(np.dot(Pt0,Pt1),dH),self.P)
    
            wOut0 = np.dot(self.P,dHtrans)
            wOut1 = y - np.dot(dH,self.wOut)
            #print("wOut vor update :",self.wOut)
            self.wOut = self.wOut + np.dot(wOut0,wOut1)
        

        
        
    
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
    
