# -*- coding: utf-8 -*-
"""
Created on Fri Nov 07 16:50:52 2014

@author: jschoenk
"""
from __future__ import division
import numpy as np

class Learner:
    def __init__(self, phiInstance):
        self.phiInst = phiInstance
        
    def learn(self, x, phiX, y, yp):
        pass
    
    def reset(self):
        pass

class SparseLearner:
    def __init__(self, phiInstance):
        self.phiInst = phiInstance
        
    def learn(self, x, phiX, idxX, y, yp):
        pass
    
    def reset(self):
        pass
    
class TreeLearner:
    def __init__(self, phiInstance):
        self.phiInst = phiInstance
        
    def learn(self, x, phiX, nodes, y, yp):
        pass
    
    def reset(self):
        pass

class AROW(Learner):
    """Adaptive Regularization Of Weights (AROW)
    
    """
    def __init__(self, phiInstance, algSetup):
        Learner.__init__(self,phiInstance)
        
        try: # use bigger r for noisy data and more stability
            self.r = algSetup["r"]
        except KeyError: #default setup
            self.r = 0.9    
        
        self.s = np.eye(phiInstance.nrAlpha) # sigma = I(matrix)
        self.phiInst = phiInstance
    
    def learn(self, x, phiX, y, yp):
        # Meaning of the variables
        # 
        # PAPER     -     CODE
        # --------------------
        # x_t       -     phix
        # m         -     m
        # v         -     v
        # sigma_t   -     s
        # mu_t      -     u
        # r         -     r
        # r lambda1 = lambda2 = 1/2r  formula(1) in paper
        #
        # read values
        r = self.r;               # tuning parameter
        s = self.s;               # sigma
        # convert x to parameter space
        # phiX = self.phiInst.aggreg(x)
        # compute the margin m
#        m = np.outer(u, phiX)
        # confidence
        v = np.dot(np.dot(phiX, s), phiX)   
        b = 1 / (v + r)
        
#        if mode == 1:     # mode = REG
        a = (y-yp) * b
        #  update mu
        deltaAlpha = np.transpose(np.dot(np.dot(a, s), phiX))
        #   update sigma
        self.s = s - np.dot(np.dot(np.dot(np.dot(b, s), phiX), phiX), s)
        #   update ILS with new parameter vector
        return deltaAlpha
    
    def reset(self):
        self.s = np.eye(self.phiInst.nrAlpha)
        
        
class CW(Learner):
    """Confidance Weighting (CW)
    
    """
    def __init__(self, phiInstance, algSetup):
        Learner.__init__(self,phiInstance)
        try:#desired pobability of correct classification
            self.p = algSetup["p"]
        except KeyError:
            self.p = 0.7
        self.sigma = np.identity(phiInstance.nrAlpha) # sigma = I(marix)
    
    def learn(self, x, phiX, y, yp):
#        # convert x to parameter space
#        # phiX = self.phiInst.aggreg(x)
#        alpha = self.phiInst.alpha
#        #compute magin
#        m = y * np.dot(phiX, alpha)
#        #compute confidence
#        v = np.dot(np.dot(phiX, self.sigma), phiX)
#        #compute amount of adaptation
#        a = max(0, (- (1+2*self.p*m) + np.sqrt( (1+2*self.p*m)**2 
#                - 8*self.p*(m-self.p*v)))/(4*self.p*v))
#            
#        #update parameter vector
#        newAlpha = alpha + np.transpose(a * y * np.dot(self.sigma, phiX))
#        #update covariance matrix        
#        self.sigma = np.linalg.inv(np.linalg.inv(self.sigma) + 2 * a * self.p * np.diag(phiX))
#        
#        return newAlpha
#        
#        if mode == 1:
        print 'CW is designed for classification.'
        
    def reset(self):
        self.sigma = np.identity(self.phiInst.nrAlpha) # sigma = I(marix)
        
class DCRLS(Learner):
    """ DeCorrelated Recursive Least Squares (DCRLS)
    
    """
    def __init__(self,phiInstance,algSetup):
        Learner.__init__(self,phiInstance)
        
        try:#covariance matrix
            self.covM = algSetup["covM"]
        except KeyError:
            self.covM = 10 ** 3
            
        self.S = self.covM * np.identity(self.phiInst.nrAlpha)
        try:#forgetting factor
            self.forgF = algSetup["forgF"]
        except KeyError:     
            self.forgF = 1
            
        self.phiInst = phiInstance
        #allocate space for matrix
        inputShape = np.zeros(self.phiInst.nrIn,dtype=int)
        for i in range(self.phiInst.nrIn):
            self.phiInst.inputs[i].init_phi_ij()
            inputShape[i] = self.phiInst.inputs[i].nrPhi
        self.N = self.phiInst.nrAlpha
        self.A = np.ones([self.N,self.N])
            
        tmpIdx = np.indices(inputShape).reshape(self.phiInst.nrIn,self.N)
        for i in range(self.N):
            for j in range(i,self.N):
                for k in range(self.phiInst.nrIn):
                    self.A[i,j] *= self.phiInst.inputs[k].phi_ij[tmpIdx[k,i],tmpIdx[k,j]]
                if i != j:
                    self.A[j,i] = self.A[i,j]
                    
        self.Ainv = np.linalg.inv(self.A)
        eigAinv = np.linalg.eig(self.Ainv)
        eigenValues = eigAinv[0]
        for i,v in enumerate(eigenValues):
            eigenValues[i] = np.sign(v)*np.sqrt(np.abs(v))
        self.Ainv = eigAinv[1]
        self.Ainv = np.dot(self.Ainv,np.diag(eigenValues))
        self.Ainv = np.dot(self.Ainv,eigAinv[1].T)

    def learn(self, x, phiX, y, yp):
        AS = np.dot(self.Ainv,self.S)
        ASA = np.dot(AS,self.Ainv)
        deltaAlpha = np.dot(ASA, phiX)/ \
        (self.forgF + np.inner(phiX, np.dot(ASA, phiX))) * (y-yp)
        
        phiTAS = np.dot(phiX.T,AS)
        self.S = self.S/self.forgF - np.outer(phiTAS, phiTAS) \
                    / (self.forgF*(self.forgF+np.inner(np.dot(phiX,ASA),phiX)))
                            
        return deltaAlpha
        
    def reset(self):
        self.S = self.covM * np.identity(self.phiInst.nrAlpha)
        
class GH(Learner):
    """Gaussian Herding (GH)
    
    """
    def __init__(self, phiInstance, algSetup):
        Learner.__init__(self,phiInstance)
        #   GH matrix variants:
        #   0 - Full
        #   1 - Exact
        #   2 - Drop
        #   3 - Project        
        try:
            self.variant = algSetup["variant"]
            #c > 0; rate of adaptation the bigger, the more adaptation is done
            self.c = algSetup["c"]
        except KeyError:
            self.variant = 0
            self.c = 1
        self.sigma = np.identity(phiInstance.nrAlpha) # sigma = I(marix)
    
    def learn(self, x, phiX, y, yp):
        #update of parameter vector is the same for all variants
        if not yp == y: 
            adjust =np.transpose(y*(max(0, 1-y*yp))/ np.dot((np.dot( \
            np.dot(phiX,self.sigma), phiX)+1/self.c),np.dot(self.sigma,phiX)))

                
        #update of sigma matrix differs for the variants
        if self.variant == 0: #full
            self.sigma = self.sigma - np.dot(np.dot(np.dot(self.sigma, np.outer(phiX,phiX)),self.sigma),\
                                (self.c**2*np.dot(np.dot(phiX,self.sigma), phiX)+2*self.c))/ \
                                (1+self.c*np.dot(np.dot(phiX,self.sigma), phiX))**2
        elif self.variant == 1: #exact
            S = np.diag(self.sigma)
            for i,x in enumerate(S):
                S[i] = S[i]/((1 + self.c * phiX[i]**2 *S[i])**2)           
            self.sigma = np.diag(S)
        
        elif self.variant == 2: #drop
            S = np.diag(self.sigma)
            S2 = S
            for i,x in enumerate(S):
                S2[i] = S[i]-np.dot(S[i]*phiX[i])^2 * (np.dot(np.dot(self.c**2*phiX, self.sigma), phiX)+2*self.c)/ \
                (1+np.dot(np.dot(self.c*phiX,self.sigma), phiX))^2;
            self.sigma = np.diag(S2)
            
        elif self.variant == 3: #project
            S = np.diag(self.sigma)
            S2 = S
            for i in enumerate(S):
                S2[i] = 1/((1/S[i])+(2*self.c+np.dot(np.dot(self.c**2*phiX,self.sigma), phiX)*phiX[i]**2))
            self.sigma = np.diag(S2)
        return adjust
        
    def reset(self):
        self.sigma = np.identity(self.phiInst.nrAlpha) # sigma = I(marix)
        
class IRMA(Learner):
    """Incremental Risk Minimization Algorithm (IRMA)
    
    """
    def __init__(self, phiInstance, algSetup):
        Learner.__init__(self,phiInstance)
        try:
            self.s = algSetup["s"]
        except KeyError:
            self.s = 0.1            
        self.s_t = self.s
        
        try:
            self.tau = algSetup["tau"]
        except KeyError:
            self.tau = 0.0
            
        try:
            self.variant = algSetup["variant"]
        except KeyError:
            self.variant = 1
            
        if self.variant == 3:
            try:
                self.maxStiff = algSetup["maxStiff"]
            except KeyError:
                print "Error: No maximal Stiffness is given!" 
        #allocate space for matrix
        inputShape = np.zeros(self.phiInst.nrIn,dtype=int)
        for i in range(self.phiInst.nrIn):
            self.phiInst.inputs[i].init_phi_ij()
            inputShape[i] = self.phiInst.inputs[i].nrPhi
        self.N = self.phiInst.nrAlpha
        self.A = np.ones([self.N,self.N])
            
        tmpIdx = np.indices(inputShape).reshape(self.phiInst.nrIn,self.N)
        for i in range(self.N):
            for j in range(i,self.N):
                for k in range(self.phiInst.nrIn):
                    self.A[i,j] *= self.phiInst.inputs[k].phi_ij[tmpIdx[k,i],tmpIdx[k,j]]
                if i != j:
                    self.A[j,i] = self.A[i,j]
                    
        self.Ainv = np.linalg.inv(self.A)
                            
    def learn(self, x, phiX, y, yp):
        # convert x to parameter Space
        # phiX = self.phiInst.aggreg(x)
        #generate B Matrix of IRMA
        deltaAlpha = np.dot(self.Ainv,phiX)*(y-yp)/ \
                       (self.s_t + np.dot(np.dot(phiX,self.Ainv), phiX))

        if self.variant == 1:
            self.s_t = self.s_t + self.tau
        elif self.variant == 2:
            self.s_t = self.s_t * self.tau
        else:
            self.s_t = 0.5*(1+ np.tanh((-np.log(1/(self.s_t/self.maxstiff)-1)+self.tau)/2))*self.maxstiff
            
        return deltaAlpha
    
    def reset(self):
        self.s_t = self.s
        
class IRMAd(Learner):
    """Incremental Risk Minimization Algorithm diagonalized (IRMAd)
    
    """
    def __init__(self, phiInstance, algSetup):
        Learner.__init__(self,phiInstance)
        try:
            self.s = algSetup["s"]
        except KeyError:
            self.s = 0.1
        self.s_t = self.s
        try:
            self.tau = algSetup["tau"]
        except KeyError:
            self.tau = 0
        try:
            self.variant = algSetup["variant"]
        except KeyError:
            self.variant = 1
        if self.variant == 3:
            try:
                self.maxStiff = algSetup["maxStiff"]
            except KeyError:
                print "Error: No maximal Stiffness is given!"
        try:
            self.diagonal = algSetup["diagonal"]
        except KeyError:
            self.diagonal = 0

            
        self.phiInst = phiInstance
        #allocate space for matrix
        self.N = self.phiInst.nrAlpha
        self.A = np.zeros([self.N, self.N])
        #TODO domain anpassen
        xref = np.linspace(-10,10,num=50)
        diffx = xref[1]-xref[0]
        dim = self.phiInst.nrIn
       
        # IF AGGREGATE == CONCAT:
        '''
        if dim <=2:
            for i in range(self.N):
                print i
                for j in range(self.N):
                    if dim == 1:
                        for x in xref:
                            phiX = self.phiInst.aggreg([x])
                            self.A[i,j] = self.A[i,j] + np.dot(phiX[i],phiX[j])*diffx
        #                    self.A = np.trapz( np.outer(self.phiInst.inputs[i](x), self.phiInst.inputs[j](x)),dx = diffx )
                            
                    elif dim == 2:
                        for x1 in xref:
                            for x2 in xref:        
                                phiX = self.phiInst.aggreg([x1,x2])
                                self.A[i,j] = self.A[i,j] + np.dot(phiX[i],phiX[j])*diffx*diffx

        '''
        # ELIF AGGREGATION == PROD???
        if dim >0:        
            self.A = self.A +1        
            dimsize = len(self.phiInst.phiforIRMA(0,0))        
            tmp = np.zeros([dim, dimsize, dimsize])        
            for k in range(dim):        
                for x in xref:
                    phiX = self.phiInst.phiforIRMA(x,k)        
                    tmp[k,:,:] = tmp[k,:,:] + np.outer(phiX,phiX) * diffx        
                m = dimsize ** (dim-k)        
                t = dimsize ** (dim-k-1)       
                for i in range(self.N):        
                    for j in range(self.N):        
                        self.A[i,j] = self.A[i,j] * tmp[k,int(np.mod(i,m)/t), int(np.mod(j,m)/t)]
        self.diagonalize()
        
    def diagonalize(self):
        if self.diagonal >= 0:
            self.Ainv = np.linalg.inv(self.A)
            # DROP
            if self.diagonal > 0:
                self.Ainv = self.d(self.Ainv)
        # PROJECT
        else:
            self.A = self.d(self.A)
            self.Ainv = np.linalg.inv(self.A)
            self.Ainv = self.d(self.Ainv)
            
    def d(self, M):
        Mhelp = np.zeros(M.shape)
        for i in range(self.N):
            j = 0
            while j <= i and j < np.absolute(self.diagonal):
                Mhelp[i-j][i] = M[i-j][i]
                Mhelp[i][i-j] = M[i][i-j]
                j += 1
        return Mhelp
                 
    def learn(self, x, phiX, y, yp, mode):
        alpha = self.phiInst.alpha
        # Ainv * phiX
        
        newAlpha = alpha.transpose() + np.inner(self.Ainv, phiX) * (y-yp)/(self.s_t +np.inner(np.inner(phiX, self.Ainv), phiX))        
    
        if self.variant == 1:
            self.s_t = self.s_t + self.tau
        elif self.variant == 2:
            self.s_t = self.s_t * self.tau
        else:
            self.s_t = 0.5*(1+ np.tanh((-np.log(1/(self.s_t/self.maxstiff)-1)+self.tau)/2))*self.maxstiff
            
        return newAlpha.transpose()
        
        if mode == 2:
            print "WARNING: IRMA is currently only designed for regression."
    
    def reset(self):
        self.s_t = self.s
        
class PA(Learner):
    """Passive Aggressive (PA)
    
    """
    def __init__(self,phiInstance, algSetup):
        Learner.__init__(self,phiInstance)
#      PA variants:
#      0 - basic PA
#      1 - basic PA-I
#      2 - basic PA-II
        try:
            self.variant = algSetup["variant"]
        except KeyError:
            self.variant = 0
        try:
            self.aggressiveness = algSetup["aggressiveness"]
        except KeyError:
            self.aggressiveness = 1
            
    def learn(self, x, phiX, y, yp):
        error = y - yp
        norm = np.sum(phiX**2)
        deltaAlpha = 0
        if(self.variant == 0):
            deltaAlpha = phiX*error/norm*self.aggressiveness
        elif(self.variant == 1):
            deltaAlpha = phiX*max(-self.aggressiveness, \
                           min(self.aggressiveness, error))/norm
        else:
            deltaAlpha = phiX*error/(norm+(1/(2*self.aggressiveness)))
#        print "phiX: ",phiX
#        print "deltaAlpha: ",deltaAlpha
        return deltaAlpha
            
    def reset(self):
        pass
    
class Perceptron(Learner):
    """Perceptron
    
    """
    def __init__(self, phiInstance, algSetup):
          Learner.__init__(self,phiInstance)
        
    def learn(self, x, phiX, y, yp):
        #get normalization for gradient step size
        #nor = np.sum(phiX**2)
        
        #update parameter vector
        deltaAlpha = phiX*(y-yp)

        return deltaAlpha
        
    def reset(self):
        pass
    
class RLS(Learner):
    """Recursive Least Squares
    
    """
    def __init__(self,phiInstance,algSetup):
        Learner.__init__(self,phiInstance)
        
        try:#covariance matrix
            self.covM = algSetup["covM"]
        except KeyError:
            self.covM = 10 ** 3
            
        self.S = self.covM * np.identity(self.phiInst.nrAlpha)
        try:#forgetting factor
            self.forgF = algSetup["forgF"]
        except KeyError:     
            self.forgF = 1

        
    def learn(self, x, phiX, y, yp):
        deltaAlpha = np.dot(self.S, phiX)/ \
        (self.forgF + np.inner(phiX, np.dot(self.S, phiX))) * (y-yp)
        
        self.S = self.S/self.forgF - np.outer(np.dot(self.S, phiX), np.dot(phiX, self.S)) \
                    / (self.forgF*(self.forgF+np.inner(np.dot(phiX,self.S),phiX)))
                            
        return deltaAlpha
        
    def reset(self):
        self.S = self.covM * np.identity(self.phiInst.nrAlpha)
        
        
class SIRMA(Learner):
    """Second order Incremental Risk Minimization Algorithm (SIRMA)
    
    """
    def __init__(self, phiInstance, algSetup):
        Learner.__init__(self,phiInstance)
        try:
            self.s = algSetup["s"]
        except KeyError:
            self.s = 0.1
        try:
            self.tau = algSetup["tau"]
        except KeyError:
            self.tau = 0.2
        try:
            self.variant = algSetup["variant"]
        except KeyError:
            self.variant = 1
        if self.variant == 3:
            try:
                self.maxStiff = algSetup["maxStiff"]
            except KeyError:
                print "Error: No maximal Stiffness is given!" 
            
        self.phiInst = phiInstance
        #allocate space for matrix
        inputShape = np.zeros(self.phiInst.nrIn,dtype=int)
        for i in range(self.phiInst.nrIn):
            self.phiInst.inputs[i].init_phi_ij()
            self.phiInst.inputs[i].init_phi_ijm()
            inputShape[i] = self.phiInst.inputs[i].nrPhi
        self.N = self.phiInst.nrAlpha
        self.A = np.ones([self.N,self.N])
            
        tmpIdx = np.indices(inputShape).reshape(self.phiInst.nrIn,self.N)
        for i in range(self.N):
            for j in range(i,self.N):
                for k in range(self.phiInst.nrIn):
                    self.A[i,j] *= self.phiInst.inputs[k].phi_ij[tmpIdx[k,i],tmpIdx[k,j]]
                if i != j:
                    self.A[j,i] = self.A[i,j]
        
        self.Am = []
        for m in range(self.N):
            self.Am.append(np.ones([self.N,self.N]))
            for i in range(self.N):
                for j in range(i,self.N):
                    for k in range(self.phiInst.nrIn):
                        self.Am[m][i,j] *= self.phiInst.inputs[k].phi_ijm[tmpIdx[k,i],tmpIdx[k,j],tmpIdx[k,m]]
                    if i != j:
                        self.Am[m][j,i] = self.Am[m][i,j]
        self.Phi_m = np.zeros([self.N])
        
    
    def learn(self, x, phiX, y, yp):
        #generate A_t Matrix of SIRMA
        A_t = self.A*self.s
        for i in range(self.N):
            A_t += self.tau*self.Phi_m[i]*self.Am[i]
        #Second order weighting matrix A_t^-1
        A_t_inv = np.linalg.inv(A_t)
        #Parameter update gain
        gain = np.dot(A_t_inv,phiX)
        #Parameter adjustment
        deltaAlpha = gain*(y-yp) / (1+np.inner(gain,phiX))

        #update density estimation
        norm = 1/np.sum(np.abs(phiX))
        for i in range(self.N):
            self.Phi_m[i] += np.abs(phiX[i])*norm
            
        return deltaAlpha
            
                    
    def reset(self):
        self.Phi_m = np.zeros([self.N])
        
class SparsePA(SparseLearner):
    
    def __init__(self,phiInstance, algSetup):
        SparseLearner.__init__(self,phiInstance)
#      PA variants:
#      0 - basic PA
#      1 - basic PA-I
#      2 - basic PA-II
        try:
            self.variant = algSetup["variant"]
        except KeyError:
            self.variant = 0
        try:
            self.aggressiveness = np.array(algSetup["aggressiveness"])
        except KeyError:
            self.aggressiveness = np.ones(1)
            
    def learn(self, x, phiX, idxX, y, yp):
        error = y - yp
        norm = np.sum(phiX**2)
        if(self.variant == 0):
            deltaAlpha = phiX*error/norm*self.aggressiveness
        elif(self.variant == 1):
            deltaAlpha = phiX*max(-self.aggressiveness, \
                           min(self.aggressiveness, error))/norm
        else:
            deltaAlpha = phiX*error/(norm+(1/(2*self.aggressiveness)))
        return np.array(deltaAlpha)
        
class TreePA(TreeLearner):
    
    def __init__(self,phiInstance, algSetup):
        TreeLearner.__init__(self,phiInstance)
        try:
            self.aggressiveness = np.array(algSetup["aggressiveness"])
        except KeyError:
            self.aggressiveness = np.ones(1)
            
    def learn(self, x, phiX, nodes, y, yp):
        error = y - yp
        norm = np.sum(phiX**2)
        return np.array(phiX*error/norm*self.aggressiveness)
        
class LRLS(Learner):
    """Local Recursive Least Squares
    
    """
    def __init__(self, phiInstance, algSetup):
        Learner.__init__(self, phiInstance)
        self.S = np.zeros(phiInstance.nrAlpha)
        
    def learn(self, x, phiX, y, yp):
        error = y - yp
        grad = error * phiX / np.sum(phiX**2)
        delta = grad * phiX
        delta /= (self.S + phiX) + (phiX==0.0)
        self.S += phiX
        return delta
    
    def reset(self):
        self.S = np.zeros(np.shape(self.S))

class SparseLRLS(SparseLearner):
    """Local Recursive Least Squares
    
    """
    def __init__(self,phiInstance, algSetup):
        SparseLearner.__init__(self,phiInstance)
        self.S = np.zeros(phiInstance.nrAlpha)
        
    def learn(self, x, phiX, idxX, y, yp):
        error = y - yp
        grad = error*phiX/np.sum(phiX**2)
        delta = (grad*phiX)/(self.S[idxX]+phiX)
        self.S[idxX] += phiX
#        print "idxX:",idxX,"self.S:",self.S[idxX]
        return delta
    
    def reset(self):
        self.S = np.zeros(np.shape(self.S))
        
class TreeLRLS(TreeLearner):
    """Local Recursive Least Squares
    
    """
    def __init__(self,phiInstance, algSetup):
        TreeLearner.__init__(self,phiInstance)
        try:
            self.defaultC = np.array(algSetup["initC"])
        except KeyError:
            self.defaultC = np.zeros(1)
        self.reset()
        
    def reset(self):
        self.phiInst.alpha.defaults["C"] = self.defaultC
        
    def learn(self, x, phiX, nodes, y, yp):
        # Get current inverse adaptation rate C
        current_C = np.zeros(np.shape(phiX))
        for i in xrange(len(phiX)):
            current_C[i] = nodes[i].C

        error = y - yp
        grad = error*phiX/np.sum(phiX**2)
        delta = (grad*phiX)/(current_C+phiX)
        
        for i in xrange(len(phiX)):
            nodes[i].C += phiX[i]
            
        return delta
        
class TreevSGD(TreeLearner):
    """Local Recursive Least Squares
    
    """
    def __init__(self,phiInstance, algSetup = {}):
        TreeLearner.__init__(self,phiInstance)
        self.reset()
        
    def reset(self):
        self.phiInst.alpha.defaults["C"] = np.zeros(1)
        self.phiInst.alpha.defaults["grad"] = np.zeros(1)
        self.phiInst.alpha.defaults["std"] = np.ones(1)
        self.phiInst.alpha.defaults["sqrGrad"] = np.zeros(1)
        self.phiInst.alpha.defaults["tau"] = np.ones(1)
        
    def learn(self, x, phiX, nodes, y, yp, rate = 1.0):
        # Get current tracking values
        current_C = np.zeros(np.shape(phiX))
        current_grad = np.zeros(np.shape(phiX))
        current_std = np.zeros(np.shape(phiX))
        current_sqrGrad = np.zeros(np.shape(phiX))
        current_tau = np.zeros(np.shape(phiX))
        for i in xrange(len(phiX)):
            current_C[i] = nodes[i].C
            current_grad[i] = nodes[i].grad
            current_std[i] = nodes[i].std
            current_sqrGrad[i] = nodes[i].sqrGrad
            current_tau[i] = nodes[i].tau

        error = (y - yp)
        grad = error*phiX/np.sum(phiX**2)
        sqrGrad = grad**2
                
        std = abs(grad)
        
        a = 1/current_tau
        
        next_grad = (1-a) * current_grad + a*grad
        next_sqrGrad = (1-a) * current_sqrGrad + a*sqrGrad
        next_std = (1-a) * current_std + a*std
        
        eta = next_grad**2/(next_sqrGrad+(next_grad==0))
        delta = eta*grad

        
        next_tau = (1-next_grad**2/next_sqrGrad)*current_tau + phiX
        next_C = current_C + phiX

        for i in xrange(len(phiX)):
            nodes[i].grad = next_grad[i]
            nodes[i].sqrGrad = next_sqrGrad[i]
            nodes[i].std = next_std[i]
            if current_C[i]<(len(x)+1):
                nodes[i].tau += 1
            else:
                if next_tau[i]<1.0 or np.isnan(next_tau[i]):
                    next_tau[i] = 1.0
                    
                nodes[i].tau = next_tau[i]
                if next_tau[i] <= 1.00001:
                    next_C[i] = 0.0
            nodes[i].C = next_C[i]
            
        return delta
        
class RLSTD:
    """Recursive Least Squares Temporal Difference Algorithm
    
    from Bradtke and Barto [1996]
    
    Bradtke, Steven J., and Andrew G. Barto. "Linear least-squares algorithms 
    for temporal difference learning." Machine Learning 22.1-3 (1996): 33-57.
    
    """
    def __init__(self,phiInstance,algSetup):
        self.phiInst = phiInstance
        
        try:#covariance matrix
            self.covM = algSetup["covM"]
        except KeyError:
            self.covM = 10.0 ** 3
            
        self.S = self.covM * np.identity(self.phiInst.nrAlpha)
        try:#forgetting factor
            self.forgF = algSetup["forgF"]
        except KeyError:     
            self.forgF = 1.0
            
        try:
            self.gamma = algSetup["gamma"]
        except KeyError:
            self.gamma = 1.0

        
    def learn(self, phiX_0, phiX_1, r_1):
        phiD = phiX_0-self.gamma*phiX_1
        phiS = np.dot(self.S,phiX_0)
        phiDS = np.dot(phiD,self.S)
        norm = 1+np.inner(phiD,phiS)
        error = r_1-np.inner(phiD,self.phiInst.alpha)
        deltaAlpha = phiS*error/norm
        self.S -= np.outer(phiS,phiDS)/norm
        return deltaAlpha
        
    def reset(self):
        self.S = self.covM * np.identity(self.phiInst.nrAlpha)