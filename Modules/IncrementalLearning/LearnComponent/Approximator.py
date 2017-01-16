# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:57:40 2015

@author: JSCHOENK
"""
from __future__ import division
import numpy as np
from scipy import integrate

from Supplementary import InputContainer
from Supplementary import SparseInputContainer

from Supplementary import ParameterContainer
from Supplementary import SparseParameterContainer
from Supplementary import TreeParameterContainer

from Supplementary import LIPApproximator
from Supplementary import SparseLIPApproximator
from Supplementary import TreeLIPApproximator
            
class TensorExpansion(ParameterContainer, LIPApproximator):
    """Linear In the Parameters approximator with tensor aggregation for multiple inputs
    
    """
    def __init__(self, foot):
        self.inputs = InputContainer(foot)
        if self.inputs.nrIn == 1:
            self.aggreg = self.inputs[0].__call__
            self.aggregD = self.inputs[0].evalD
        nrAlpha = len(self.aggreg(np.zeros(self.inputs.nrIn)))
        ParameterContainer.__init__(self, foot, nrAlpha)
        LIPApproximator.__init__(self)
        
                  
    def aggreg(self, x):
        out = np.array([1.])
        for i in xrange(self.inputs.nrIn):
            outtmp = self.inputs[i](x[i])
            out = np.outer(out, outtmp).flatten()
        return out
        
    def aggregD(self, x):
        out = [np.array([1.])]*self.inputs.nrIn
#        print "out init:", out
#        print ""
        for i in xrange(1,self.inputs.nrIn):
            outtmp = self.inputs[i](x[i])
            out[i] = np.outer(out[i-1],outtmp).flatten()
#        print "out prep:", out
#        print ""
        for i in xrange(self.inputs.nrIn):
            out[i] = np.outer(out[i],self.inputs[i].evalD(x[i])).flatten()
#        print "out D:", out
#        print ""
        tmp = np.array([1.])
        for i in xrange(1,self.inputs.nrIn):
#            print "here"
            tmp = np.outer(self.inputs[-i](x[-i]),tmp).flatten()
            out[-i-1] = np.outer(out[-i-1],tmp).flatten()
#        print "out final:", out
#        print ""
        return out
        

class Concatenation(ParameterContainer, LIPApproximator):
    """Linear In the Parameters approximator    
    
    """
    def __init__(self, foot):
        self.inputs = InputContainer(foot)
        nrAlpha = len(self.aggreg(np.zeros(self.inputs.nrIn)))
        ParameterContainer.__init__(self, foot, nrAlpha)
        LIPApproximator.__init__(self)
            
    def aggreg(self, x):
        out = []
        for i in xrange(self.inputs.nrIn):
            a = self.inputs[i](x[i])
            out.append(a)
        return np.hstack(out)
        
            
class DecorrelatedTensorExpansion(ParameterContainer, LIPApproximator):
    """Linear In the Parameters approximator    
    
    """
    def __init__(self, foot):
        self.inputs = InputContainer(foot)
        
        inputShape = np.zeros(self.inputs.nrIn, dtype=int)
        for i in xrange(self.inputs.nrIn):
            self.inputs[i].init_phi_ij()
            inputShape[i] = self.inputs[i].nrPhi
            
        ParameterContainer.__init__(self, foot, inputShape.prod())
        LIPApproximator.__init__(self)
        
        self.A = np.ones([self.nrAlpha, self.nrAlpha])
            
        tmpIdx = np.indices(inputShape).reshape(self.inputs.nrIn, self.nrAlpha)
        for i in xrange(self.nrAlpha):
            for j in xrange(i,self.nrAlpha):
                for k in xrange(self.inputs.nrIn):
                    self.A[i ,j] *= self.inputs[k].phi_ij[tmpIdx[k, i],tmpIdx[k, j]]
                if i != j:
                    self.A[j, i] = self.A[i, j]
                    
        self.Ainv = np.linalg.inv(self.A)
        eigAinv = np.linalg.eig(self.Ainv)
        eigenValues = eigAinv[0]
        for i, v in enumerate(eigenValues):
            eigenValues[i] = np.sign(v)*np.sqrt(np.abs(v))
        self.Ainv = eigAinv[1]
        self.Ainv = np.dot(self.Ainv,np.diag(eigenValues))
        self.Ainv = np.dot(self.Ainv,eigAinv[1].T)
        
    def aggreg(self, x):
        tmp = np.array([1.])
        for i in xrange(self.inputs.nrIn):
             outtmp = self.inputs[i](x[i])
             tmp = np.outer(tmp, outtmp).flatten()
        
        out = np.dot(self.Ainv, tmp).flatten()
        return out

    def aggerd1D(self, x):
        tmp = self.inputs[0](x).flatten()
        out = np.dot(self.Ainv,tmp)
        return out
        
class LocalModelTensorExpansion(ParameterContainer, LIPApproximator):
    def __init__(self, foot):
        self.inputs = InputContainer(foot)
        
        inputShape = np.zeros(self.inputs.nrIn, dtype=int)
        self.degree = foot["degree"]
        for i in xrange(self.inputs.nrIn):
            self.inputs[i].init_phi_ij()
            inputShape[i] = self.inputs[i].nrPhi
        nrParam = inputShape.prod()*(1+self.inputs.nrIn*self.degree)
        ParameterContainer.__init__(self, foot, nrParam)
        
        LIPApproximator.__init__(self)
        
    def aggreg(self, x):
        out = np.array([1.])
        model = np.array([1.])
        for i in xrange(self.degree):
            model = np.hstack([model, x**(i+1)])
        for i in xrange(self.inputs.nrIn):
            outtmp = self.inputs[i](x[i])
            out = np.outer(out, outtmp).flatten()
        out = np.outer(out,model).flatten()
        return out
        
    def aggregD(self, x):
        loc0 = np.array([1.])
        for i in xrange(self.inputs.nrIn):
            outtmp = self.inputs[i](x[i])
            loc0 = np.outer(loc0, outtmp).flatten()
        locD = [np.array([1.])]*self.inputs.nrIn
        
        
        model0 = np.ones(1)
        modelD = [np.zeros(1)]*self.inputs.nrIn
        for i in xrange(self.degree):
            model0 = np.hstack([model0, x**(i+1)])
            for j in xrange(self.inputs.nrIn):
                stack = x**(i+1)
                stack[j] = (i+1)*x[j]**i
#                print np.shape(modelD),np.shape(stack)
                modelD[j] = np.hstack([modelD[j], stack])
            
        for i in xrange(1,self.inputs.nrIn):
            outtmp = self.inputs[i](x[i])
            locD[i] = np.outer(locD[i-1],outtmp).flatten()

        for i in xrange(self.inputs.nrIn):
            locD[i] = np.outer(locD[i],self.inputs[i].evalD(x[i])).flatten()

        tmp = np.array([1.])
        for i in xrange(1,self.inputs.nrIn):
            tmp = np.outer(self.inputs[-i](x[-i]),tmp).flatten()
            locD[-i-1] = np.outer(locD[-i-1],tmp).flatten()

        out = []
        for i in xrange(self.inputs.nrIn):
#            print np.shape(loc0), np.shape(modelD[i])
            A = np.outer(loc0,modelD[i]).flatten()
            B = np.outer(locD[i],model0).flatten()
            out.append(A+B)
        
        return out
    
            
class SparseTensorExpansion(SparseParameterContainer, SparseLIPApproximator):
    """Sparse Linear In the Parameters approximator with tensor aggregation for multiple inputs
    
    """
    def __init__(self, foot):
        self.inputs = SparseInputContainer(foot)
        nrAlpha = 1
        for i in xrange(self.inputs.nrIn):
            nrAlpha *= self.inputs[i].nrPhi
        SparseParameterContainer.__init__(self, foot, nrAlpha)
        SparseLIPApproximator.__init__(self)
            
        # calculate multiDimCoord2oneDimCoord multipliers
        self.multipliers = np.zeros(self.inputs.nrIn, dtype=int)
        self.multipliers[-1] = 1
        for i in xrange(self.inputs.nrIn-1, 0, -1):
            self.multipliers[i-1] = self.multipliers[i] * self.inputs[i].nrPhi
        
    def aggreg(self, x):
        out = np.array([1.])
        idx = np.array([0])
        for i in xrange(self.inputs.nrIn):
             out_in_i,idx_in_i = self.inputs[i](x[i])
             tmp = []
             for j in idx_in_i:
                 tmp.append(idx+self.multipliers[i]*j)
             idx = np.hstack(tmp)
             out = np.outer(out, out_in_i).flatten()
        nonzero = np.nonzero(out)
        out = out[nonzero]
        idx = idx[nonzero]
        return out, idx
                
class SparseLinearSimplicialBSpline(SparseParameterContainer, SparseLIPApproximator):
    """ First order simplicial B-Spline approximator
    
    """
    def __init__(self, foot):
        self.inputs = SparseInputContainer(foot)
        nrAlpha = 1
        for i in xrange(self.inputs.nrIn):
            nrAlpha *= self.inputs[i].nrPhi
        SparseParameterContainer.__init__(self, foot, nrAlpha)
        SparseLIPApproximator.__init__(self)
            
        # calculate multiDimCoord2oneDimCoord multipliers
        self.multipliers = np.zeros(self.inputs.nrIn, dtype=int)
        self.multipliers[-1] = 1
        for i in xrange(self.inputs.nrIn-1, 0, -1):
            self.multipliers[i-1] = self.multipliers[i] * self.inputs[i].nrPhi
            
    def aggreg(self, x):
        # Evaluate the membership functions of every single input dimension
        # to find the cell index and local normalised cell coordinate.
        cell_base_index = np.zeros(self.inputs.nrIn)
        cell_dim_coord = np.zeros(self.inputs.nrIn)
        cell_orientation = np.zeros(self.inputs.nrIn)
        for i in xrange(self.inputs.nrIn):
            tmp_eval, tmp_index = self.inputs[i](x[i])
            if len(tmp_index) == 2:
                # Here we distinguish between even and odd nodes in order to
                # form a symetric subdivision of the regular hypercube grid.
                # Only even nodes are allowed to be the base index of a cell.
                if 0x1 & tmp_index[0]: #odd
                    cell_base_index[i] = tmp_index[1] # Thus, here we consider
                    # the top next node instead of the odd one found here.
                    cell_dim_coord[i] = tmp_eval[0]
                    cell_orientation[i] = -1
                else: # even
                    cell_base_index[i] = tmp_index[0]
                    cell_dim_coord[i] = tmp_eval[1]
                    cell_orientation[i] = 1
            else:
                cell_base_index[i] = tmp_index[0]
                cell_dim_coord[i] = 0
                
        numel = np.count_nonzero(cell_dim_coord) + 1
        if numel==1:
            return np.array([1.0]), [int(np.inner(cell_base_index,self.multipliers))]
        
        # Sort local normalized cell coordinates in order to determine the
        # current simplex for evaluation.
        sort_perm = np.argsort(cell_dim_coord)[::-1]

        # Form the indicies of the corners of the current simplex based on the
        # cell base index and the permutation for sorting the local normalized
        # coordinetes.
        index_set = np.zeros([numel, self.inputs.nrIn])
        index_set[0] = cell_base_index.copy()
        for i in xrange(numel - 1):
            cell_base_index[sort_perm[i]] += cell_orientation[sort_perm[i]]
            index_set[i + 1] = cell_base_index.copy()
        
        # Reduce multi dimensional index set to one dimensional index set
        tmp_index_set = np.zeros(numel, dtype=int)
        for i in xrange(numel):
            for j in xrange(self.inputs.nrIn):
                tmp_index_set[i] += index_set[i, j]*self.multipliers[j]
        
        # Evaluation of the base functions of the current simplex.
        phi = np.zeros([numel])
        phi[0] = 1 - cell_dim_coord[sort_perm[0]]
        phi[numel - 1] = cell_dim_coord[sort_perm[numel - 2]]
        for i in xrange(1, numel - 1):
             phi[i] = cell_dim_coord[sort_perm[i - 1]] - cell_dim_coord[sort_perm[i]]
        nonZero = np.nonzero(phi)
        phi = phi[nonZero]
        index_set = tmp_index_set[nonZero]
        return phi, index_set
        
class AdaptiveLinearSimplicialBSpline(TreeParameterContainer, TreeLIPApproximator):
    """ First order simplicial B-Spline approximator
    
    """
    def __init__(self, foot):
        self.inputs = SparseInputContainer(foot)
        if self.inputs.nrIn == 1:
            self.aggreg = self.inputs[0].__call__
        TreeParameterContainer.__init__(self, foot)
        TreeLIPApproximator.__init__(self)
        
            
    def aggreg(self, x):
        # Evaluate the membership functions of every single input dimension
        # to find the cell index and local normalised cell coordinate.
        nrIn = self.inputs.nrIn
        cell_base_index = np.zeros(nrIn)
        cell_dim_coord = np.zeros(nrIn)
        cell_orientation = np.zeros(nrIn)
        numel = 1
        for i in xrange(nrIn):
            tmp_eval, tmp_index = self.inputs[i](x[i])
            if len(tmp_index) == 2:
                numel +=1
                # Here we distinguish between even and odd nodes in order to
                # form a symetric subdivision of the regular hypercube grid.
                # Only even nodes are allowed to be the base index of a cell.
                if 0b1 & tmp_index[0]: #odd
                    cell_base_index[i] = tmp_index[1] # Thus, here we consider
                    # the top next node instead of the odd one found here.
                    cell_dim_coord[i] = tmp_eval[0]
                    cell_orientation[i] = -1
                else: # even
                    cell_base_index[i] = tmp_index[0]
                    cell_dim_coord[i] = tmp_eval[1]
                    cell_orientation[i] = 1
            else:
                
                if 0b1 & tmp_index[0]: #odd
                    numel += 1
                    cell_base_index[i] = tmp_index[0]-1
                    cell_dim_coord[i] = 1.0
                    cell_orientation[i] = 1
                else:
                    cell_base_index[i] = tmp_index[0]
                    cell_dim_coord[i] = 0.0
                    cell_orientation[i] = 0
                    
        if numel==1:
            return np.array([1.0]), cell_base_index.reshape([1,nrIn])
        
        # Sort local normalized cell coordinates in order to determine the
        # current simplex for evaluation.
        sort_perm = np.argsort(cell_dim_coord)[::-1]

        # Form the indicies of the corners of the current simplex based on the
        # cell base index and the permutation for sorting the local normalized
        # coordinetes.
        index_set = np.zeros([numel, nrIn])
        index_set[0] = cell_base_index.copy()
        for i in xrange(numel - 1):
            cell_base_index[sort_perm[i]] += cell_orientation[sort_perm[i]]
            index_set[i + 1] = cell_base_index.copy()
                
        # Evaluation of the base functions of the current simplex.
        phi = np.zeros([numel])
        phi[0] = 1 - cell_dim_coord[sort_perm[0]]
        phi[numel - 1] = cell_dim_coord[sort_perm[numel - 2]]
        for i in xrange(1, numel - 1):
             phi[i] = cell_dim_coord[sort_perm[i - 1]] - cell_dim_coord[sort_perm[i]]
        nonZero = np.nonzero(phi)
        phi = np.array(phi[nonZero])
        index_set = index_set[nonZero]
        return phi, index_set
        
class AdaptiveCubicSimplicialBSpline(TreeParameterContainer, TreeLIPApproximator):
    """ First order simplicial B-Spline approximator
    
    """
    def __init__(self, foot):
        self.inputs = SparseInputContainer(foot)
        if self.inputs.nrIn == 1:
            self.aggreg = self.aggreg1D
        TreeParameterContainer.__init__(self, foot)
        TreeLIPApproximator.__init__(self)
        
    def cubic1D(self, x):
        return (2*x+1)*(x-1)**2
        
            
    def aggreg1D(self, x):
        phiX, idxX = self.inputs[0](x)
        if len(phiX) == 2:
            for i in xrange(len(phiX)):
                phiX[i] = self.cubic1D(1-phiX[i])
        return phiX, idxX
        
    def aggreg(self, x):
        # Evaluate the membership functions of every single input dimension
        # to find the cell index and local normalised cell coordinate.
        nrIn = self.inputs.nrIn
        cell_base_index = np.zeros(nrIn)
        cell_dim_coord = np.zeros(nrIn)
        cell_orientation = np.zeros(nrIn)
        numel = 1
        skip = 0
        cut = 0
        for i in xrange(nrIn):
            tmp_eval, tmp_index = self.inputs[i](x[i])
            if len(tmp_index) == 2:
                numel +=1
                # Here we distinguish between even and odd nodes in order to
                # form a symetric subdivision of the regular hypercube grid.
                # Only even nodes are allowed to be the base index of a cell.
                if 0b1 & tmp_index[0]: #odd
                    cell_base_index[i] = tmp_index[1] # Thus, here we consider
                    # the top next node instead of the odd one found here.
                    cell_dim_coord[i] = tmp_eval[0]
                    cell_orientation[i] = -1
                else: # even
                    cell_base_index[i] = tmp_index[0]
                    cell_dim_coord[i] = tmp_eval[1]
                    cell_orientation[i] = 1
            else:
                if 0b1 & tmp_index[0]: #odd
                    skip += 1
                    cell_base_index[i] = tmp_index[0]-1
                    cell_dim_coord[i] = 1.0
                    cell_orientation[i] = 1
                else:
                    cut += 1
                    cell_base_index[i] = tmp_index[0]
                    cell_dim_coord[i] = 0.0
                    cell_orientation[i] = 0
                
        
        if numel==1:
            cell_base_index += cell_dim_coord
            return np.ones(1), cell_base_index.reshape([1,nrIn])


        # Sort local normalized cell coordinates in order to determine the
        # current simplex for evaluation.
        sort_perm = np.argsort(cell_dim_coord)
        sort_perm = sort_perm[::-1]
        for i in xrange(skip):
            cell_base_index[sort_perm[i]] += cell_orientation[sort_perm[i]]
        skip_perm = sort_perm[skip:nrIn-cut]
        
        # residual coordinate
        r = cell_dim_coord[skip_perm]
        

        # Form the indicies of the corners of the current simplex based on the
        # cell base index and the permutation for sorting the local normalized
        # coordinetes.
        index_set = np.zeros([numel, nrIn])
        index_set[0] = cell_base_index.copy()

        for i in xrange(numel-1):
            cell_base_index[skip_perm[i]] += cell_orientation[skip_perm[i]]
            index_set[i + 1] = cell_base_index.copy()
            
        
        # Evaluation of the base functions of the current simplex.    
        
#        phi = np.zeros([numel])
        phi = np.ones([numel])
        for i in r:
            phi[0] *= self.cubic1D(i)
            phi[-1] *= self.cubic1D(1-i)
    
        for i in xrange(1,numel-1):
            a = self.cubic1D(r[i-1])/(self.cubic1D(r[i-1])+self.cubic1D(1-r[i]))
            b = np.array(1.0-a)
            for j in xrange(0,i):
                a *= self.cubic1D((1-r[j]))
                b *= self.cubic1D((1-r[j])/(1-r[i]))
            for j in xrange(i,numel-1):
                a *= self.cubic1D((r[j])/(r[i-1]))
                b *= self.cubic1D((r[j]))
            phi[i] = a+b
            
        nonZero = np.nonzero(phi)
        phi = np.array(phi[nonZero])
#        phi /= phi.sum()
        index_set = index_set[nonZero]
        return phi, index_set
