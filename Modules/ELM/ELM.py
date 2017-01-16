# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 17:49:03 2016

@author: Henning
"""
from __future__ import division
from Modules.IncrementalLearning.IncrementalLearningSystem import IncrementalLearningSystem 
import numpy as np
import time
from ActivationFunctions import Sigmoid,TanH

class ELM ( IncrementalLearningSystem ):
    
    def init ( self , foot ): 
        self.foot= foot
        IncrementalLearningSystem.init( self , foot )
        
    def prepare ( self , antecessor ): 
        self.nrIn = len(antecessor["xLearn"].output) + 1
        self.nrHidden = 7
        self.wIn = np.random.randn(self.nrHidden,self.nrIn)
        #self.wIn = np.ones((self.nrHidden,self.nrIn))
        self.wOut = np.random.randn(self.nrHidden,1)
        self.nodes = np.empty(self.nrHidden,dtype=object)
        for ni,_ in enumerate(self.nodes):
            #zufaelliges SIgmoid erstellen
            sig = TanH()
            #sig = self.TanH()
            self.nodes[ni] = sig
        self.H = None
        #from paper: nr of initial trainingdata not less than nr of hidden nodes
        self.nrInitT = self.nrHidden + 30
        #self.nrInitT = 70080
        self.initT = []
        self.counter = 0
        print("PREPARE ENDE")
    def normalize(self, x):
        #print("x: ",x)
        x = x/3.14
        if isinstance(x, np.float64):
            x = [x,1]
        else:
            xn = np.empty(x.size+1)
            xn[0:-1] = x
            xn[-1] = 1
            x = xn
        x = np.array(x)
        return x
        #xd(t+1) = xd(t) * (t/(t+1)) + X * (1/(t+1))

            
            
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
            self.P = np.linalg.inv(toInv)
            print("P nach init: ",self.P)
            
            #print("shapes: ",self.P.shape," ",self.H.shape," ",initT.shape)
            #print(np.dot(np.dot(self.P,np.transpose(self.H)),initT).shape)
            
            #wOut nach online in batch phase:
            #self.wOut = np.dot(np.dot(self.P,np.transpose(self.H)),initT)
            
            print("Iself.wOut: ",self.wOut)            
            #print("initt: ",np.shape(initT))
            #print("initT: ",self.initT)
            #time.sleep(5)
        #sonst: (one by one version) 
        elif sizeH >self.nrInitT:
            #print("learn online")
            dHtrans = np.transpose(dH)
            #print("shapes: P: ",self.P.shape,"dH: ",dH.shape,"dHtrans: ",dHtrans.shape)
            #print("Vself.P: ",self.P)        
            
            Pt0 = np.dot(np.dot(np.dot(self.P,dHtrans),dH),self.P)
            Pt1 = np.float64(1.0 + np.dot(np.dot(dH,self.P),dHtrans))
            self.P = self.P - np.true_divide(Pt0, Pt1)
            #self.P = self.P + np.identity(len(self.P))
            #print("Nself.P: ",self.P)  
            #print("Vself.P: ",np.shape(self.P))
            #print("VdH: ",np.shape(dH))
            wOut0 = np.dot(self.P,dHtrans)
            #wOut0 = wOut0.reshape((self.nrHidden,1))
#            print("wOut0.shape: ",np.shape(wOut0))
            wOut1 = np.float64(y - np.dot(dH,self.wOut)) * 0.001
            #wOut1 = np.sign(wOut1) * 0.0001
#            print("wOut0: ",wOut0)
#            print("wOut1: ",wOut1)
#            print("wOut: ",self.wOut)
            #time.sleep(3)
            #print("wOut1: ",wOut1," mit: y= ",y," und: ",np.dot(dH,self.wOut))
            #print("Vself.wOut: ",np.shape(self.wOut))
            self.wOut = self.wOut + wOut0 * wOut1
            #print("Nself.wOut: ",self.wOut)
            #time.sleep(30)
        #print("eval nach lernen: ",self.evaluate(x))
#        if sizeH > 20:
#            raise ArithmeticError("first learned")
        self.counter+=1
        if self.counter == 400:
            print("wOut am Ende: ",self.wOut)
            
if __name__ == "__main__":
    sig = ELM.Sigmoid(-0.4,1.6)
    print(sig.evaluate(3))
    
