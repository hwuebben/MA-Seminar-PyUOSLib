"""
Created on Mon Feb 16 13:58:40 2015

@author: jschoenk
"""
from __future__ import division
import numpy as np
import pylab as plt
from scipy import integrate
#from scipy.stats import norm

class NodalBaseFnc:
    """ Nodal Basis Function
    
    """
    def __init__(self,foot):
        try:
            self.nodes = np.array(foot["nodes"])
        except KeyError:
            self.nodes = np.linspace(0,1,7)
            
        try:
            self.domain = np.array(foot["domain"])
        except KeyError:
            self.domain = np.array([min(self.nodes),max(self.nodes)])
        
        self.nrPhi = len(self.nodes)

class CPPolynom(NodalBaseFnc):
    """Continous Piecewise Polynomial

    """    
    def __init__(self,foot):
        NodalBaseFnc.__init__(self,foot)
        self.degree = foot["degree"]
        self.nrPhi = np.size(self.nodes)*(self.degree+1)-self.degree        
        
    def __call__(self, x):
        out_local = np.zeros(np.shape(self.nodes))
        if (x<=self.nodes[0]):
            out_local[0] = 1
        elif (x>=self.nodes[-1]):
            out_local[-1] = 1
        else:
            idx = np.argmax(x<self.nodes)
            out_local[idx] = (x-self.nodes[idx-1])/(self.nodes[idx]-self.nodes[idx-1])
            out_local[idx-1] = 1-out_local[idx]
        out_global = np.power(x, range(self.degree+1))
        out = np.outer(out_local,out_global).flatten()[:self.nrPhi]
        
        return out
        
        
class Fourier:
    """Discrete Fourier Series
    
    """
    def __init__(self,foot):
        try:
            self.degree = foot["degree"]
        except KeyError:
            self.degree = 2
        self.nrPhi = self.degree*2 + 1
        
        try:
            self.domain = np.array(foot["domain"])
        except KeyError:
            self.domain = np.array([-10,10])
        
    def __call__(self, x):
        x_in = (x-self.domain[0])/(self.domain[1]-self.domain[0])
        x_in = (x_in-0.5)*2*np.pi
        out = np.zeros([self.nrPhi])
        out[0] = 1
        for i in range(self.degree):
            out[1+i] = np.sin((i+1)*x_in)
            out[1+self.degree+i] = np.cos((i+1)*x_in)
        return out
        
    def evalD(self,x):
        out = np.zeros(self.nrPhi)
        for i in range(self.nrPhi):
            out[i] = i * (x**(np.max([i-1,0])))
        return out
    
    def init_phi_ij(self):
        if not hasattr(self,"phi_ij"):
            self.phi_ij = np.eye(self.nrPhi)
                
    def init_phi_ijm(self):
        if not hasattr(self,"phi_ijm"):
            self.phi_ijm = np.zeros([self.nrPhi,self.nrPhi,self.nrPhi])
            for m in range(self.nrPhi):
                for i in range(self.nrPhi):
                    for j in range(self.nrPhi):
                        tmp = lambda x: self(x)[i]*self(x)[j]*self(x)[m]
                        self.phi_ijm[i,j,m] = integrate.quad(tmp,self.domain[0],self.domain[1])[0]
                        self.phi_ijm[j,i,m] = self.phi_ijm[i,j,m]

class GLTgauss(NodalBaseFnc):
    """Grid-base Look-up Table (GLT) with gaussian interpolation
    
    """
    def __init__(self,foot):
        NodalBaseFnc.__init__(self,foot)
        self.distance = np.diff(self.nodes)
        self.scale = 1 / (2*np.sqrt(-np.log(1/2)))
        
    def gauss(self, x, u, sigma):
        return np.exp( -(x-u)**2 / (2*sigma**2) )
        
    def __call__(self, x):
        out = np.zeros(np.shape(self.nodes))
        if self.nrPhi==1:
            out = np.array([1])
        else:
            if (x<=self.nodes[0]):
                out[0] = 1
            else:
                out[0] = self.gauss(x,self.nodes[0], self.distance[0]*self.scale)
                
            for i in range(1,self.nrPhi-1):
                u = self.nodes[i]
                if u < x:
                    out[i] = self.gauss(x,u, self.distance[i]*self.scale)
                else:
                    out[i] = self.gauss(x,u, self.distance[i-1]*self.scale)
            
            if (x>=self.nodes[-1]):
                out[-1] = 1
            else:
                out[-1] = self.gauss(x,self.nodes[-1], self.distance[-1]*self.scale)
            out = out / np.sum(out)

        return out
    
    
    def init_phi_ij(self):
        if not hasattr(self,"phi_ij"):
            self.phi_ij = np.zeros([self.nrPhi,self.nrPhi])
            for i in range(self.nrPhi):
                for j in range(i,self.nrPhi):
                    tmp = lambda x: self(x)[i]*self(x)[j]
                    self.phi_ij[i,j] = integrate.quad(tmp,self.domain[0],self.domain[1])[0]
                    self.phi_ij[j,i] = self.phi_ij[i,j]

    def init_phi_ijm(self):
        if not hasattr(self,"phi_ijm"):
            self.phi_ijm = np.zeros([self.nrPhi,self.nrPhi,self.nrPhi])
            for m in range(self.nrPhi):
                for i in range(self.nrPhi):
                    for j in range(self.nrPhi):
                        tmp = lambda x: self(x)[i]*self(x)[j]*self(x)[m]
                        self.phi_ijm[i,j,m] = integrate.quad(tmp,self.domain[0],self.domain[1])[0]
                        self.phi_ijm[j,i,m] = self.phi_ijm[i,j,m]

class GLTlinear(NodalBaseFnc):
    """Grid-base Look-up Table (GLT) with linear interpolation
    
    """
    def __init__(self, foot):
        NodalBaseFnc.__init__(self,foot)
    
    def __call__(self, x):
        out = np.zeros(np.shape(self.nodes))
        if (x<=self.nodes[0]):
            out[0] = 1
        elif (x>=self.nodes[-1]):
            out[-1] = 1
        else:
            idx = np.argmax(x<self.nodes)
            out[idx] = (x-self.nodes[idx-1])/(self.nodes[idx]-self.nodes[idx-1])
            out[idx-1] = 1-out[idx]
        return out
        
    def evalD(self,x):
        out = np.zeros(np.shape(self.nodes))
        if (x<=self.nodes[0]):
            out[0] = 0
        elif (x>=self.nodes[-1]):
            out[-1] = 0
        else:
            idx = np.argmax(x<self.nodes)
            out[idx] = 1/(self.nodes[idx]-self.nodes[idx-1])
            out[idx-1] = -out[idx]
        return out
        
    def init_phi_ij(self):
        if not hasattr(self,"phi_ij"):
            self.phi_ij = np.zeros([self.nrPhi,self.nrPhi])
            for i in range(self.nrPhi):
                for j in range(i,self.nrPhi):
                    tmp = lambda x: self(x)[i]*self(x)[j]
                    self.phi_ij[i,j] = integrate.quad(tmp,self.domain[0],self.domain[1])[0]
                    self.phi_ij[j,i] = self.phi_ij[i,j]

    def init_phi_ijm(self):
        if not hasattr(self,"phi_ijm"):
            self.phi_ijm = np.zeros([self.nrPhi,self.nrPhi,self.nrPhi])
            for m in range(self.nrPhi):
                for i in range(self.nrPhi):
                    for j in range(i,self.nrPhi):
                        tmp = lambda x: self(x)[i]*self(x)[j]*self(x)[m]
                        self.phi_ijm[i,j,m] = integrate.quad(tmp,self.domain[0],self.domain[1])[0]
                        self.phi_ijm[j,i,m] = self.phi_ijm[i,j,m]
                        
class GLTlinearONB(NodalBaseFnc):
    """Grid-base Look-up Table (GLT) with linear interpolation and OrthoNormal Basis
    
    """
    def __init__(self, foot):
        NodalBaseFnc.__init__(self,foot)
        
        self.scaling = []
        for i in range(self.nrPhi):
            scale_i = np.zeros([i+1])
            scale_i[-1] = 1
            
            for j in range(i):
                tmp = lambda x: self(x)[i]*self(x)[j]
                scale_i[j] = -integrate.quad(tmp,self.domain[0],self.domain[1])[0]            
            self.scaling.append(scale_i)
            tmp = lambda x: self(x)[i]*self(x)[i]
            norm = np.sqrt(integrate.quad(tmp,self.domain[0],self.domain[1])[0])
            self.scaling[i] /= norm
    
    def __call__(self, x):
        out = np.zeros(np.shape(self.nodes))
        if (x<=self.nodes[0]):
            out[0] = 1
        elif (x>=self.nodes[-1]):
            out[-1] = 1
        else:
            idx = np.argmax(x<self.nodes)
            out[idx] = (x-self.nodes[idx-1])/(self.nodes[idx]-self.nodes[idx-1])
            out[idx-1] = 1-out[idx]
        for i,scales in enumerate(self.scaling):
            out[i] = np.inner(out[:(i+1)],scales)
        return out
        
    def evalD(self,x):
        out = np.zeros(np.shape(self.nodes))
        if (x<=self.nodes[0]):
            out[0] = 0
        elif (x>=self.nodes[-1]):
            out[-1] = 0
        else:
            idx = np.argmax(x<self.nodes)
            out[idx] = 1/(self.nodes[idx]-self.nodes[idx-1])
            out[idx-1] = -out[idx]
        return out
        
    def init_phi_ij(self):
        if not hasattr(self,"phi_ij"):
            self.phi_ij = np.zeros([self.nrPhi,self.nrPhi])
            for i in range(self.nrPhi):
                for j in range(i,self.nrPhi):
                    tmp = lambda x: self(x)[i]*self(x)[j]
                    self.phi_ij[i,j] = integrate.quad(tmp,self.domain[0],self.domain[1])[0]
                    self.phi_ij[j,i] = self.phi_ij[i,j]

    def init_phi_ijm(self):
        if not hasattr(self,"phi_ijm"):
            self.phi_ijm = np.zeros([self.nrPhi,self.nrPhi,self.nrPhi])
            for m in range(self.nrPhi):
                for i in range(self.nrPhi):
                    for j in range(i,self.nrPhi):
                        tmp = lambda x: self(x)[i]*self(x)[j]*self(x)[m]
                        self.phi_ijm[i,j,m] = integrate.quad(tmp,self.domain[0],self.domain[1])[0]
                        self.phi_ijm[j,i,m] = self.phi_ijm[i,j,m]


class Lagrange(NodalBaseFnc):
    """Lagrange polynomials
    
    """
    def __init__(self,foot):
        NodalBaseFnc.__init__(self,foot)
        
    def __call__(self, x):
        out = np.ones([self.nrPhi])
        for i in range(self.nrPhi):
            for j in range(self.nrPhi):
                if i is not j:
                    out[i] *= (x-self.nodes[j])/(self.nodes[i]-self.nodes[j])
        return out
        
    def evalD(self,x):
        out = np.zeros(self.nrPhi)
        for i in range(self.nrPhi):
            out[i] = i * (x**(np.max([i-1,0])))
        return out
    
    def init_phi_ij(self):
        if not hasattr(self,"phi_ij"):
            self.phi_ij = np.zeros([self.nrPhi,self.nrPhi])
            for i in range(self.nrPhi):
                for j in range(i,self.nrPhi):
                    tmp = lambda x: self(x)[i]*self(x)[j]
                    self.phi_ij[i,j] = integrate.quad(tmp,self.domain[0],self.domain[1])[0]
                    self.phi_ij[j,i] = self.phi_ij[i,j]
                
    def init_phi_ijm(self):
        if not hasattr(self,"phi_ijm"):
            self.phi_ijm = np.zeros([self.nrPhi,self.nrPhi,self.nrPhi])
            for m in range(self.nrPhi):
                for i in range(self.nrPhi):
                    for j in range(self.nrPhi):
                        tmp = lambda x: self(x)[i]*self(x)[j]*self(x)[m]
                        self.phi_ijm[i,j,m] = integrate.quad(tmp,self.domain[0],self.domain[1])[0]
                        self.phi_ijm[j,i,m] = self.phi_ijm[i,j,m]

class Legendre:
    """Legendre polynomials
    
    """
    def __init__(self,foot):
        try:
            self.degree = foot["degree"]
        except KeyError:
            self.degree = 2
        self.nrPhi = self.degree + 1
        
        try:
            self.domain = np.array(foot["domain"])
        except KeyError:
            self.domain = np.array([0,1])

    def __call__(self, x):
        x_in = (x-self.domain[0])/(self.domain[1]-self.domain[0])
        x_in = x_in*2-1
        out = np.zeros([self.nrPhi])
        out[0] = 1
        out[1] = x_in
        for i in range(2,self.nrPhi):
            n = i-1
            out[i] = ((2*n+1)*x_in*out[n]-n*out[n-1])/i
        return out
        
    def evalD(self,x):
        out = np.zeros(self.nrPhi)
        for i in range(self.nrPhi):
            out[i] = i * (x**(np.max([i-1,0])))
        return out
    
    def init_phi_ij(self):
        if not hasattr(self,"phi_ij"):
            self.phi_ij = np.eye(self.nrPhi)
                
    def init_phi_ijm(self):
        if not hasattr(self,"phi_ijm"):
            self.phi_ijm = np.zeros([self.nrPhi,self.nrPhi,self.nrPhi])
            for m in range(self.nrPhi):
                for i in range(self.nrPhi):
                    for j in range(self.nrPhi):
                        tmp = lambda x: self(x)[i]*self(x)[j]*self(x)[m]
                        self.phi_ijm[i,j,m] = integrate.quad(tmp,self.domain[0],self.domain[1])[0]
                        self.phi_ijm[j,i,m] = self.phi_ijm[i,j,m]
                        
class polynom:
    """Polynomial
    
    """
    def __init__(self,foot):
        try:
            self.degree = foot["degree"]
        except KeyError:
            self.degree = 2
        self.nrPhi = self.degree + 1
        
        try:
            self.domain = np.array(foot["domain"])
        except KeyError:
            self.domain = np.array([0,1])
        
    def __call__(self, x):
        out = np.power(x, range(self.degree+1))
        return out
        
    def evalD(self,x):
        out = np.zeros(self.nrPhi)
        for i in range(self.nrPhi):
            out[i] = i * (x**(np.max([i-1,0])))
        return out
    
    def init_phi_ij(self):
        if not hasattr(self,"phi_ij"):
            xmin = float(self.domain[0])
            xmax = float(self.domain[1])
            self.phi_ij = np.zeros([self.nrPhi,self.nrPhi])
            for i in range(self.nrPhi):
                for j in range(i,self.nrPhi):
                    self.phi_ij[i,j] = 1/(i+j+1) * xmax**(i+j+1)-1/(i+j+1) * xmin**(i+j+1)
                    self.phi_ij[j,i] = self.phi_ij[i,j]
                
    def init_phi_ijm(self):
        if not hasattr(self,"phi_ijm"):
            xmin = float(self.domain[0])
            xmax = float(self.domain[1])
            self.phi_ijm = np.zeros([self.nrPhi,self.nrPhi,self.nrPhi])
            for m in range(self.nrPhi):
                for i in range(self.nrPhi):
                    for j in range(self.nrPhi):
                        tmp = lambda x: (x**(i+j)*np.abs(x)**m)/np.sum(np.power(np.abs(x), range(self.degree+1)))
                        self.phi_ijm[i,j,m] = integrate.quad(tmp,xmin,xmax)[0]
                        self.phi_ijm[j,i,m] = self.phi_ijm[i,j,m]


class SparseGLTlinear(NodalBaseFnc):

    def __init__(self, foot):
        NodalBaseFnc.__init__(self,foot)
    
    def __call__(self, x):
        if (x<=self.nodes[0]):
            out = np.array([1])
            idx = [0]
        elif (x>=self.nodes[-1]):
            out = np.array([1])
            idx = [self.nrPhi-1]
        else:
            idx = np.argmax(x<self.nodes)
            out = np.zeros([2])
            out[1] = (x-self.nodes[idx-1])/(self.nodes[idx]-self.nodes[idx-1])
            out[0] = 1-out[1]
            idx = [idx-1,idx]
        return out,idx
        
    def init_phi_ij(self):
        if not hasattr(self,"phi_ij"):
            self.phi_ij = np.zeros([self.nrPhi,self.nrPhi])
            for i in range(self.nrPhi):
                for j in range(i,self.nrPhi):
                    tmp = lambda x: self(x)[i]*self(x)[j]
                    self.phi_ij[i,j] = integrate.quad(tmp,self.domain[0],self.domain[1])[0]
                    self.phi_ij[j,i] = self.phi_ij[i,j]

    def init_phi_ijm(self):
        if not hasattr(self,"phi_ijm"):
            self.phi_ijm = np.zeros([self.nrPhi,self.nrPhi,self.nrPhi])
            for m in range(self.nrPhi):
                for i in range(self.nrPhi):
                    for j in range(i,self.nrPhi):
                        tmp = lambda x: self(x)[i]*self(x)[j]*self(x)[m]
                        self.phi_ijm[i,j,m] = integrate.quad(tmp,self.domain[0],self.domain[1])[0]
                        self.phi_ijm[j,i,m] = self.phi_ijm[i,j,m]
                        
class SparseUIGLTlinear():
    """Sparse Uniform Implicit Grid-base Look-up Table with linear interpolation
    
    """
    def __init__(self,foot):
        try:
            self.res = np.array(foot["res"])
        except KeyError:
            self.res = np.ones(1)
        try:
            self.offset = np.array(foot["offset"])
        except KeyError:
            self.offset = np.zeros(1)
            
    def __call__(self, x):
        x_shift = x + self.offset
        ratio = x_shift/self.res
        index = int(np.ceil(ratio))
        phi = index-ratio
        return [phi,1-phi],[index-1,index]