 -*- coding: utf-8 -*-
"""
Created on Thu Jan 05 02:43:33 2017

@author: Henning
"""
from __future__ import division
from Modules.IncrementalLearning.IncrementalLearningSystem import IncrementalLearningSystem
import numpy as np

class ELM_Batch(IncrementalLearningSystem):
    
    def init ( self , foot ): 
        IncrementalLearningSystem.init( self , foot )
        
    def prepare ( self , antecessor ): 
        self.n_hidden_units = 20
        
    def learn(self, X,labels):
        X = np.column_stack([X,np.ones([X.shape[0],1])])
        self.random_weights = np.random.randn(X.shape[1],self.n_hidden_units)
        G = np.tanh(X.dot(self.random_weights))
        self.w_elm = np.linalg.pinv(G).dot(labels)

    def evaluate(self, X):
        X = np.column_stack([X,np.ones([X.shape[0],1])])
        G = np.tanh(X.dot(self.random_weights))
        return G.dot(self.w_elm)