# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:57:40 2015

@author: JSCHOENK
"""
from __future__ import division
import numpy as np

from MPyUOSLib import BasicProcessingModule

from LearnComponent.Supplementary import LinearSimplicialBSplineLayer
from LearnComponent.Supplementary import CubicSimplicialBSplineLayer
from LearnComponent.Supplementary import MonomialLayer

class IncrementalLearningSystem(BasicProcessingModule):
    """
    Generic frame for your incremental learning system!
    """
    def __init__(self, foot, default = {}):
        """
        Setup all the parameters you need
        """
        BasicProcessingModule.__init__(self, foot, default)
        
    def __call__(self, xLearn, yLearn, xEval=None, index=0):
        """
        Evaluate and learn in a proper way
        """
        if xEval is None:
            out = self.eval_and_learn(xLearn, yLearn)    
        else:
            out = self.evaluate(xEval)
            self.learn(xLearn, yLearn)
        return out
    
    def eval_and_learn(self, xLearn, yLearn):
        
        out = self.evaluate(xLearn)
        self.learn(xLearn, yLearn)
        return out
        
class CompoundLearningSystem(IncrementalLearningSystem):
    """ Compound Incremental Learning System
    
    # Adaptive MISO system
    # MISO: Multiple Input Single Output
    # Mapping f: R^n -> R
    #
    # Contains:
    # -Approximator
    # -Learning algorithm
    #
    # Provides:
    # -Evaluation of approximator (__call__)
    # -Training of approximator (learn, evalAndLearn)
    """
    def __init__(self, foot):
        IncrementalLearningSystem.__init__(self, foot)
        # setup approximator
        self.approximator = []
        try:
            approx_input = foot["approximator"]
        except KeyError:
            approx_input = {}
        try:
            self.approxType = foot["approximator"]["kind"]
        except KeyError:
            self.approxType = "TensorExpansion"
        exec("from LearnComponent.Approximator import " + self.approxType)
        exec("self.approximator = " + self.approxType + "(approx_input)")
        
        # setup learner
        self.learner = []
        try:
            learner_input = foot["learner"]
        except KeyError:
            learner_input = {}
        try:
            self.learnerType = foot["learner"]["name"]
        except KeyError:
            self.learnerType = "PA"
        exec("from LearnComponent.Learner import " + self.learnerType)
        exec("self.learner = " + self.learnerType + "(self.approximator,learner_input)")
        
        
    def evaluate(self, x):
        # Evaluation of approximator
        #
        return self.approximator(x)
            
    def get_alpha(self):
        return self.approximator.alpha.copy()
        
    def learn(self, x, y):
        phiX = self.approximator.aggreg(x)
        yp = self.approximator.evaluatePhiX(phiX)
        self.approximator.alpha += self.learner.learn(x, phiX, y, yp)

    def eval_and_learn(self, x, y):
        phiX = self.approximator.aggreg(x)
        yp = self.approximator.evaluatePhiX(phiX)
        self.approximator.alpha += self.learner.learn(x, phiX, y, yp)
        return yp
        
    def reset(self):
        self.approximator.reset()
        self.learner.reset()
        
class SparseCompoundLearningSystem(IncrementalLearningSystem):
    # Incremental Learning System (ILS)
    # Adaptive MISO system
    # MISO: Multiple Input Single Output
    # Mapping f: R^n -> R
    #
    # Contains:
    # -Approximator
    # -Learning algorithm
    #
    # Provides:
    # -Evaluation of approximator (__call__)
    # -Training of approximator (learn, evalAndLearn)

    def __init__(self, foot):
        IncrementalLearningSystem.__init__(self, foot)
        # setup approximator
        self.approximator = []
        try:
            approx_input = foot["approximator"]
        except KeyError:
            approx_input = {}
        try:
            self.approxType = foot["approximator"]["kind"]
        except KeyError:
            self.approxType = "TensorExpansion"
        exec("from LearnComponent.Approximator import Sparse" + self.approxType)
        exec("self.approximator = Sparse" + self.approxType + "(approx_input)")
        
        # setup learner
        self.learner = []
        try:
            learner_input = foot["learner"]
        except KeyError:
            learner_input = {}
        try:
            self.learnerType = foot["learner"]["name"]
        except KeyError:
            self.learnerType = "PA"
        exec("from LearnComponent.Learner import Sparse" + self.learnerType)
        exec("self.learner = Sparse" + self.learnerType + "(self.approximator,learner_input)")
        
        
    def evaluate(self, x):
        # Evaluation of approximator
        #
        return self.approximator(x)
            
    def get_alpha(self):
        return self.approximator.alpha.copy()
        
    def learn(self, x, y):
        phiX, idxX = self.approximator.aggreg(x)
        yp = self.approximator.evaluatePhiX(phiX, idxX)
        self.approximator.alpha[idxX] += self.learner.learn(x, phiX, idxX, y, yp)

    def eval_and_learn(self ,x, y):
        phiX, idxX = self.approximator.aggreg(x)
        yp = self.approximator.evaluatePhiX(phiX, idxX)
        self.approximator.alpha[idxX] += self.learner.learn(x, phiX, idxX, y, yp)
        return yp
        
    def reset(self):
        self.approximator.reset()
        self.learner.reset()
        
class AdaptiveLinearSimplicialBSpline(IncrementalLearningSystem):
    """Adaptive Linear Simplicial B-Spline Learning System
    
    """
    def __init__(self, foot):
        IncrementalLearningSystem.__init__(self,foot)
        self.approximator = []
        try:
            approx_input = foot["approximator"]
        except KeyError:
            approx_input = {}
        try:
            self.approxType = foot["approximator"]["kind"]
        except KeyError:
            self.approxType = "AdaptiveLinearSimplicialBSpline"
        exec("from LearnComponent.Approximator import " + self.approxType)
        exec("self.approximator = " + self.approxType + "(approx_input)")
        
        # setup learner
        self.learner = []
        try:
            learner_input = foot["learner"]
        except KeyError:
            learner_input = {}
        try:
            self.learnerType = foot["learner"]["name"]
        except KeyError:
            self.learnerType = "PA"
        exec("from LearnComponent.Learner import Tree" + self.learnerType)
        exec("self.learner = Tree" + self.learnerType + "(self.approximator,learner_input)")
    
        
    def evaluate(self, x):
        # Evaluation of approximator
        #
        return self.approximator(x)
                
    def learn(self, x, y):
        phiX, idxX = self.approximator.aggreg(x)
        yp = self.approximator.evaluatePhiX(phiX,idxX)
        nodes = self.approximator.alpha[idxX]
        for i in xrange(len(nodes)):
            if nodes[i] == None:
                nodes[i] = self.approximator.alpha.insert(idxX[i])
        deltaAlpha = self.learner.learn(x,phiX,nodes,y,yp)
        for i in xrange(len(nodes)):
            nodes[i].value += deltaAlpha[i]

    def eval_and_learn(self, x, y):
        phiX, idxX = self.approximator.aggreg(x)
        nodes = self.approximator.alpha[idxX]
        for i in xrange(len(nodes)):
            if nodes[i] == None:
                nodes[i] = self.approximator.alpha.insert(idxX[i])
            if nodes[i] == None:
                print idxX[i]
                nodes[i] = self.approximator.alpha[idxX[i]]
        yp = self.approximator.evaluatePhiX(phiX,idxX)
        deltaAlpha = self.learner.learn(x, phiX, nodes, y, yp)
        for i in xrange(len(nodes)):
            nodes[i].value += deltaAlpha[i]
        return yp
        
    def reset(self):
        self.approximator.reset()
        self.learner.reset()
        
class AdaptiveCubicSimplicialBSpline(IncrementalLearningSystem):
    """Adaptive Linear Simplicial B-Spline Learning System
    
    """
    def __init__(self, foot):
        IncrementalLearningSystem.__init__(self,foot)
        self.approximator = []
        try:
            approx_input = foot["approximator"]
        except KeyError:
            approx_input = {}
        try:
            self.approxType = foot["approximator"]["kind"]
        except KeyError:
            self.approxType = "AdaptiveCubicSimplicialBSpline"
        exec("from LearnComponent.Approximator import " + self.approxType)
        exec("self.approximator = " + self.approxType + "(approx_input)")
        
        # setup learner
        self.learner = []
        try:
            learner_input = foot["learner"]
        except KeyError:
            learner_input = {}
        try:
            self.learnerType = foot["learner"]["name"]
        except KeyError:
            self.learnerType = "PA"
        exec("from LearnComponent.Learner import Tree" + self.learnerType)
        exec("self.learner = Tree" + self.learnerType + "(self.approximator,learner_input)")
        
    def evaluate(self, x):
        # Evaluation of approximator
        #
        return self.approximator(x)
                
    def learn(self, x, y):
        phiX, idxX = self.approximator.aggreg(x)
        yp = self.approximator.evaluatePhiX(phiX,idxX)
        nodes = self.approximator.alpha[idxX]
        for i in xrange(len(nodes)):
            if nodes[i] == None:
                nodes[i] = self.approximator.alpha.insert(idxX[i])
        deltaAlpha = self.learner.learn(x,phiX,nodes,y,yp)
        for i in xrange(len(nodes)):
            nodes[i].value += deltaAlpha[i]

    def eval_and_learn(self, x, y):
        phiX, idxX = self.approximator.aggreg(x)
        nodes = self.approximator.alpha[idxX]
        for i in xrange(len(nodes)):
            if nodes[i] == None:
                nodes[i] = self.approximator.alpha.insert(idxX[i])
        yp = self.approximator.evaluatePhiX(phiX,idxX)
        deltaAlpha = self.learner.learn(x, phiX, nodes, y, yp)
        for i in xrange(len(nodes)):
            nodes[i].value += deltaAlpha[i]
        return yp
        
    def reset(self):
        self.approximator.reset()
        self.learner.reset()
        
class LayeredLinearSimplicialBSpline(IncrementalLearningSystem):
    """ Layered Linear Simplicial B-Splines
    
    """
    def __init__(self, foot):
        IncrementalLearningSystem.__init__(self,foot)
        self.delta = foot["delta"]
        try:
            self.initRes = foot["res"]
        except KeyError:
            self.initRes = 1.0
            
        try:
            self.initOffset = foot["offset"]
        except KeyError:
            self.initOffset = 0.0
            
    def prepare(self, antecessor):
        self.nrIn = int(np.size(antecessor["xLearn"].output))
        self.nrLayer = 1
        self.layer = []
        self.layer.append(LinearSimplicialBSplineLayer(self.nrIn, self.initRes, self.initOffset, self.delta))
        
    def evaluate(self, x):
        out = self.layer[0].evaluate(x)
        if self.nrLayer>1:
            layer_index = 1
            go_to_next_layer = True
            while go_to_next_layer and layer_index<self.nrLayer:
                layer_approx = self.layer[layer_index].approximator
                phiX, idxX = layer_approx.aggreg(x)
                nodes = layer_approx.alpha[idxX]
                go_to_next_layer = False
                for n in nodes:
                    if type(n) is not None:
                        go_to_next_layer = True
                if go_to_next_layer:
                    out += layer_approx.evaluatePhiX(phiX,idxX)
                layer_index += 1
        return out
        
    def learn(self, x, y):
        y_residual = y
        notInRange, expansion_direction = self.getExpensionDirection(x)
        while notInRange:
            self.expandRange(expansion_direction)
            notInRange, expansion_direction = self.getExpensionDirection(x)
            
#        for i in range(self.nrLayer):            
        out_layer, layerC = self.layer[0].learn_and_eval(x,y_residual)
        y_residual -= out_layer
        layer_index = 1
        while abs(y_residual)>self.delta and layer_index<self.nrLayer:
            out_layer, layerC = self.layer[layer_index].learn_and_eval(x,y_residual)
            y_residual -= out_layer
            layer_index += 1
            
        if abs(y_residual)>self.delta:
            if layerC>(self.nrIn+1)*10 and self.nrLayer<6:
                next_res = 0.5 * self.layer[-1].res
                next_offset = self.layer[-1].offset
                self.layer.append(LinearSimplicialBSplineLayer(self.nrIn, next_res, next_offset, self.delta))
                self.layer[-1].learn(x,y_residual)
                self.nrLayer += 1
                print "Adding new layer @res", next_res, " number of layers:",self.nrLayer
#        else:
#            for i in range(layer_index,self.nrLayer):
#                self.layer[i].vanish(x)
    
    def expandRange(self, expansion_direction):
        newOffset = self.layer[0].offset.copy()
        res = self.layer[0].res
        for i in xrange(self.nrIn):
            newOffset[i] += expansion_direction[i]*res[i]
            
        new_top_layer = LinearSimplicialBSplineLayer(self.nrIn, 2*res, newOffset, self.delta)
        
        node_idx = self.layer[0].approximator.alpha.toList('multiDimKey')
        node_C = self.layer[0].approximator.alpha.toList('C')
        for i in xrange(len(node_idx)):
            current_pos = self.layer[0].approximator.inputs.idx2pos(node_idx[i])
            phiX, idxX = new_top_layer.approximator.aggreg(current_pos)
            for j in xrange(len(phiX)):
                new_top_layer.approximator.alpha[idxX[j],'C'] += phiX[j]*node_C[i]
        self.layer.insert(0,new_top_layer)
        self.nrLayer += 1
        print "Expanding Range to res",2*res
        
    def getExpensionDirection(self, x):
        notInRange = False
        idx_x_min = self.layer[0].approximator.coordTransform.minDim
        idx_x_max = self.layer[0].approximator.coordTransform.maxDim
        if not idx_x_min:
            return False, np.zeros(self.nrIn)
        expansion_direction = np.zeros(self.nrIn)
        if self.nrIn==1:
            xmin = self.layer[0].approximator.inputs[0].idx2pos(idx_x_min)
            xmax = self.layer[0].approximator.inputs[0].idx2pos(idx_x_max)
            if x<xmin:
                notInRange = True
#                expansion_direction[0] = -1
            if x>xmax:
                notInRange = True
        else:
            for i in xrange(self.nrIn):
                xmin = self.layer[0].approximator.inputs[i].idx2pos(idx_x_min[i])
                xmax = self.layer[0].approximator.inputs[i].idx2pos(idx_x_max[i])
                if x[i]<xmin:
                    notInRange = True
                    expansion_direction[i] = -1
                if x[i]>xmax:
                    notInRange = True
        return notInRange, expansion_direction
        
class LayeredCubicSimplicialBSpline(IncrementalLearningSystem):
    """ Layered Linear Simplicial B-Splines
    
    """
    def __init__(self, foot):
        IncrementalLearningSystem.__init__(self,foot)
        self.delta = foot["delta"]
        try:
            self.initRes = foot["res"]
        except KeyError:
            self.initRes = 1.0
            
        try:
            self.initOffset = foot["offset"]
        except KeyError:
            self.initOffset = 0.0
            
    def prepare(self, antecessor):
        self.nrIn = int(np.size(antecessor["xLearn"].output))
        self.nrLayer = 1
        self.layer = []
        self.layer.append(CubicSimplicialBSplineLayer(self.nrIn, self.initRes, self.initOffset, self.delta))
        
    def evaluate(self, x):
        out = self.layer[0].evaluate(x)
        if self.nrLayer>1:
            layer_index = 1
            go_to_next_layer = True
            while go_to_next_layer and layer_index<self.nrLayer:
                layer_approx = self.layer[layer_index].approximator
                phiX, idxX = layer_approx.aggreg(x)
                nodes = layer_approx.alpha[idxX]
#                print phiX,idxX,nodes
#                if len(idxX)==1:
#                    print idxX,nodes
                go_to_next_layer = False
                try:
                    if nodes is not None:
                        for n in nodes:
                            if type(n) is not type(None):
                                go_to_next_layer = True                    
                except TypeError:
                    if nodes:
                        go_to_next_layer = True
                        
                if go_to_next_layer:
                    out += layer_approx.evaluatePhiX(phiX,idxX)
                layer_index += 1
        return out
        
    def learn(self, x, y):
        y_residual = y
        notInRange, expansion_direction = self.getExpensionDirection(x)
        while notInRange:
            self.expandRange(expansion_direction)
            notInRange, expansion_direction = self.getExpensionDirection(x)
            
#        for i in range(self.nrLayer):            
        out_layer, layerC = self.layer[0].learn_and_eval(x,y_residual)
        y_residual -= out_layer
        layer_index = 1
        while abs(y_residual)>self.delta and layer_index<self.nrLayer:
            out_layer, layerC = self.layer[layer_index].learn_and_eval(x,y_residual)
            y_residual -= out_layer
            layer_index += 1
            
        if abs(y_residual)>self.delta:
            if layerC>(self.nrIn+1)*10 and self.nrLayer<6:
                next_res = 0.5 * self.layer[-1].res
                next_offset = self.layer[-1].offset
                self.layer.append(CubicSimplicialBSplineLayer(self.nrIn, next_res, next_offset, self.delta))
                self.layer[-1].learn(x,y_residual)
                self.nrLayer += 1
                print "Adding new layer @res", next_res, " number of layers:",self.nrLayer
#        else:
#            for i in range(layer_index,self.nrLayer):
#                self.layer[i].vanish(x)
    
    def expandRange(self, expansion_direction):
        newOffset = self.layer[0].offset.copy()
        res = self.layer[0].res
        for i in xrange(self.nrIn):
            newOffset[i] += expansion_direction[i]*res[i]
            
        new_top_layer = CubicSimplicialBSplineLayer(self.nrIn, 2*res, newOffset, self.delta)
        
        node_idx = self.layer[0].approximator.alpha.toList('multiDimKey')
        node_C = self.layer[0].approximator.alpha.toList('C')
        for i in xrange(len(node_idx)):
            current_pos = self.layer[0].approximator.inputs.idx2pos(node_idx[i])
            phiX, idxX = new_top_layer.approximator.aggreg(current_pos)
            for j in xrange(len(phiX)):
                new_top_layer.approximator.alpha[idxX[j],'C'] += phiX[j]*node_C[i]
        self.layer.insert(0,new_top_layer)
        self.nrLayer += 1
        print "Expanding Range to res",2*res
        
    def getExpensionDirection(self, x):
        notInRange = False
        idx_x_min = self.layer[0].approximator.coordTransform.minDim
        idx_x_max = self.layer[0].approximator.coordTransform.maxDim
        if not idx_x_min:
            return False, np.zeros(self.nrIn)
        expansion_direction = np.zeros(self.nrIn)
        if self.nrIn==1:
            xmin = self.layer[0].approximator.inputs[0].idx2pos(idx_x_min)
            xmax = self.layer[0].approximator.inputs[0].idx2pos(idx_x_max)
            if x<xmin:
                notInRange = True
#                expansion_direction[0] = -1
            if x>xmax:
                notInRange = True
        else:
            for i in xrange(self.nrIn):
                xmin = self.layer[0].approximator.inputs[i].idx2pos(idx_x_min[i])
                xmax = self.layer[0].approximator.inputs[i].idx2pos(idx_x_max[i])
                if x[i]<xmin:
                    notInRange = True
                    expansion_direction[i] = -1
                if x[i]>xmax:
                    notInRange = True
        return notInRange, expansion_direction
        
        
class LayeredSimplexTesselationLinearInterpolation(IncrementalLearningSystem):
    """ Layered Simplicial Tesselation with linear interpolation
    
    """
    def __init__(self, foot):
        IncrementalLearningSystem.__init__(self,foot)
        self.delta = foot["delta"]
        try:
            self.initRes = foot["res"]
        except KeyError:
            self.initRes = 1.0
            
        try:
            self.initOffset = foot["offset"]
        except KeyError:
            self.initOffset = 0.0
            
    def prepare(self, antecessor):
        self.nrIn = int(np.size(antecessor["xLearn"].output))
        self.nrLayer = 1
        self.layer = []
        self.layer.append(LinearSimplicialBSplineLayer(self.nrIn, self.initRes, self.initOffset, self.delta))
        
    def evaluate(self, x):
        out = self.layer[0].evaluate(x)
        layerStd = []
        if self.nrLayer>1:
            layer_index = 1
            go_to_next_layer = True
            while go_to_next_layer and layer_index<self.nrLayer:
                layer_approx = self.layer[layer_index].approximator
                phiX, idxX = layer_approx.aggreg(x)
                nodes = layer_approx.alpha[idxX]
                evalStd = 0.0
                for i in range(len(phiX)):
                    try:
                        evalStd += phiX[i]*nodes[i].std
                    except TypeError:
                        pass
                    except AttributeError:
                        pass
                layerStd.append(evalStd)
                go_to_next_layer = False
                try:
                    if nodes is not None:
                        for n in nodes:
                            if type(n) is not type(None):
                                go_to_next_layer = True                    
                except TypeError:
                    if nodes:
                        go_to_next_layer = True
                if go_to_next_layer:
                    out += layer_approx.evaluatePhiX(phiX,idxX)
#                    if layer_index<3:
#                        out += layer_approx.evaluatePhiX(phiX,idxX)
#                    else:
#                        a0 = layerStd[0]
#                        x0 = layer_index-2
#                        y0 = layerStd[x0]
#                        if y0 != 0.0:
#                            b = (a0-y0)/(x0*y0)
#                            test_std = a0/(1+b*(layer_index-1))
#                            if layerStd[layer_index-1]<test_std:
#                                out += layer_approx.evaluatePhiX(phiX,idxX)
#                        else:
#                            out += layer_approx.evaluatePhiX(phiX,idxX)
                layer_index += 1
        return out
        
    def learn(self, x, y):
        y_residual = y
        notInRange, expansion_direction = self.getExpensionDirection(x)
        while notInRange:
            self.expandRange(expansion_direction)
            notInRange, expansion_direction = self.getExpensionDirection(x)
            

        layer_index = 0
        go_to_next_layer = True
        while go_to_next_layer and layer_index<self.nrLayer:
            layer_approx = self.layer[layer_index].approximator
            phiX, idxX = layer_approx.aggreg(x)
            nodes = layer_approx.alpha[idxX]
            go_to_next_layer = False
            try:
                if nodes is not None:
                    for n in nodes:
                        if type(n) is not type(None):
                            go_to_next_layer = True                    
            except TypeError:
                if nodes:
                    go_to_next_layer = True
            layer_index += 1
            
        LayerWeights = 0.5**(np.arange(layer_index-1,-1,-1))
        layerStd = np.zeros(layer_index)
        for i in range(layer_index):
            out_layer, layerC, evalStd = self.layer[i].learn_and_eval(x,y_residual,LayerWeights[i])
            layerStd[i] = evalStd
            y_residual -= out_layer
        
        no_noise = False
        if layer_index<3:
            no_noise = True
        else:
            a0 = layerStd[0]
            x0 = layer_index-2
            y0 = layerStd[x0]
            b = (a0-y0)/(x0*y0)
            test_std = a0/(1+b*(layer_index-1))
            no_noise = layerStd[layer_index-1]<test_std
            
        if no_noise and layerC>(self.nrIn+1)*10:
            if layer_index<self.nrLayer:
                self.layer[layer_index].learn(x,y_residual)
            elif self.nrLayer<10:
                print layerStd
                next_res = 0.5 * self.layer[-1].res
                next_offset = self.layer[-1].offset
                self.layer.append(LinearSimplicialBSplineLayer(self.nrIn, next_res, next_offset, self.delta))
                self.layer[-1].learn(x,y_residual)
                self.nrLayer += 1
                print "Adding new layer @res", next_res, " number of layers:",self.nrLayer

    
    def expandRange(self, expansion_direction):
        newOffset = self.layer[0].offset.copy()
        res = self.layer[0].res
        for i in xrange(self.nrIn):
            newOffset[i] += expansion_direction[i]*res[i]
            
        new_top_layer = LinearSimplicialBSplineLayer(self.nrIn, 2*res, newOffset, self.delta)
        
        node_idx = self.layer[0].approximator.alpha.toList('multiDimKey')
        node_C = self.layer[0].approximator.alpha.toList('C')
        for i in xrange(len(node_idx)):
            current_pos = self.layer[0].approximator.inputs.idx2pos(node_idx[i])
            phiX, idxX = new_top_layer.approximator.aggreg(current_pos)
            for j in xrange(len(phiX)):
                new_top_layer.approximator.alpha[idxX[j],'C'] += phiX[j]*node_C[i]
        self.layer.insert(0,new_top_layer)
        self.nrLayer += 1
        print "Expanding Range to res",2*res
        
    def getExpensionDirection(self, x):
        notInRange = False
        idx_x_min = self.layer[0].approximator.coordTransform.minDim
        idx_x_max = self.layer[0].approximator.coordTransform.maxDim
        if not idx_x_min:
            return False, np.zeros(self.nrIn)
        expansion_direction = np.zeros(self.nrIn)
        if self.nrIn==1:
            xmin = self.layer[0].approximator.inputs[0].idx2pos(idx_x_min)
            xmax = self.layer[0].approximator.inputs[0].idx2pos(idx_x_max)
            if x<xmin:
                notInRange = True
#                expansion_direction[0] = -1
            if x>xmax:
                notInRange = True
        else:
            for i in xrange(self.nrIn):
                xmin = self.layer[0].approximator.inputs[i].idx2pos(idx_x_min[i])
                xmax = self.layer[0].approximator.inputs[i].idx2pos(idx_x_max[i])
                if x[i]<xmin:
                    notInRange = True
                    expansion_direction[i] = -1
                if x[i]>xmax:
                    notInRange = True
        return notInRange, expansion_direction
        
class LayeredSimplexTesselationCubicInterpolation(IncrementalLearningSystem):
    """ Layered Simplicial Tesselation with linear interpolation
    
    """
    def __init__(self, foot):
        IncrementalLearningSystem.__init__(self,foot)
        if not hasattr(self,"res"):
            self.res = 1.0
        if not hasattr(self,"offset"):
            self.offset = 0.0
        if not hasattr(self,"maxNrLayer"):
            self.maxNrLayer = 5
            
    def prepare(self, antecessor):
        self.nrIn = int(np.size(antecessor["xLearn"].output))
        self.nrLayer = 1
        self.layer = []
        self.layer.append(CubicSimplicialBSplineLayer(self.nrIn, self.res, self.offset))
        
    def evaluate(self, x):
        out = self.layer[0].evaluate(x)
        layerStd = []
        if self.nrLayer>1:
            layer_index = 1
            go_to_next_layer = True
            while go_to_next_layer and layer_index<self.nrLayer:
                layer_approx = self.layer[layer_index].approximator
                phiX, idxX = layer_approx.aggreg(x)
                nodes = layer_approx.alpha[idxX]
                evalStd = 0.0
                for i in range(len(phiX)):
                    try:
                        evalStd += phiX[i]*nodes[i].std
                    except TypeError:
                        pass
                    except AttributeError:
                        pass
                layerStd.append(evalStd)
                go_to_next_layer = False
                try:
                    if nodes is not None:
                        for n in nodes:
                            if type(n) is not type(None):
                                go_to_next_layer = True                    
                except TypeError:
                    if nodes:
                        go_to_next_layer = True
                if go_to_next_layer:
                    out += layer_approx.evaluatePhiX(phiX,idxX)
                layer_index += 1
        return out
        
    def learn(self, x, y):
        y_residual = y
        notInRange, expansion_direction = self.getExpensionDirection(x)
        while notInRange:
            self.expandRange(expansion_direction)
            notInRange, expansion_direction = self.getExpensionDirection(x)
            

        layer_index = 0
        go_to_next_layer = True
        while go_to_next_layer and layer_index<self.nrLayer:
            layer_approx = self.layer[layer_index].approximator
            phiX, idxX = layer_approx.aggreg(x)
            nodes = layer_approx.alpha[idxX]
            go_to_next_layer = False
            try:
                if nodes is not None:
                    for n in nodes:
                        if type(n) is not type(None):
                            go_to_next_layer = True                    
            except TypeError:
                if nodes:
                    go_to_next_layer = True
            layer_index += 1
            
        LayerWeights = 0.5**(np.arange(layer_index-1,-1,-1))
        layerStd = np.zeros(layer_index)
        for i in range(layer_index):
            out_layer, layerC, evalStd = self.layer[i].learn_and_eval(x,y_residual,LayerWeights[i])
            layerStd[i] = evalStd
            if i>1:
                a0 = layerStd[0]
                x0 = i-1
                y0 = layerStd[x0]
                b = (a0-y0)/(x0*y0)
                test_std = a0/(1+b*(i))
                if layerStd[i]<test_std:
                    y_residual -= out_layer
                else:
                    y_residual *= 0
            else:
                y_residual -= out_layer
        
        no_noise = False
        if layer_index<3:
            no_noise = True
        else:
            a0 = layerStd[0]
            x0 = layer_index-2
            y0 = layerStd[x0]
            b = (a0-y0)/(x0*y0)
            test_std = a0/(1+b*(layer_index-1))
            no_noise = layerStd[layer_index-1]<test_std
            
        if no_noise and layerC>(self.nrIn+1)*10:
            if layer_index<self.nrLayer:
                self.layer[layer_index].learn(x,y_residual)
            elif self.nrLayer<self.maxNrLayer:
                print layerStd
                next_res = 0.5 * self.layer[-1].res
                next_offset = self.layer[-1].offset
                self.layer.append(CubicSimplicialBSplineLayer(self.nrIn, next_res, next_offset))
                self.layer[-1].learn(x,y_residual)
                self.nrLayer += 1
                print "Adding new layer @res", next_res, " number of layers:",self.nrLayer

    
    def expandRange(self, expansion_direction):
        newOffset = self.layer[0].offset.copy()
        res = self.layer[0].res
        for i in xrange(self.nrIn):
            newOffset[i] += expansion_direction[i]*res[i]
            
        new_top_layer = LinearSimplicialBSplineLayer(self.nrIn, 2*res, newOffset, self.delta)
        
        node_idx = self.layer[0].approximator.alpha.toList('multiDimKey')
        node_C = self.layer[0].approximator.alpha.toList('C')
        for i in xrange(len(node_idx)):
            current_pos = self.layer[0].approximator.inputs.idx2pos(node_idx[i])
            phiX, idxX = new_top_layer.approximator.aggreg(current_pos)
            for j in xrange(len(phiX)):
                new_top_layer.approximator.alpha[idxX[j],'C'] += phiX[j]*node_C[i]
        self.layer.insert(0,new_top_layer)
        self.nrLayer += 1
        print "Expanding Range to res",2*res
        
    def getExpensionDirection(self, x):
        notInRange = False
        idx_x_min = self.layer[0].approximator.coordTransform.minDim
        idx_x_max = self.layer[0].approximator.coordTransform.maxDim
#        print "idx_x_min",idx_x_min
        if idx_x_min is None:
            return False, np.zeros(self.nrIn)
        expansion_direction = np.zeros(self.nrIn)
        if self.nrIn==1:
            xmin = self.layer[0].approximator.inputs[0].idx2pos(idx_x_min)
            xmax = self.layer[0].approximator.inputs[0].idx2pos(idx_x_max)
            if x<xmin:
                notInRange = True
#                expansion_direction[0] = -1
            if x>xmax:
                notInRange = True
        else:
            for i in xrange(self.nrIn):
                xmin = self.layer[0].approximator.inputs[i].idx2pos(idx_x_min[i])
                xmax = self.layer[0].approximator.inputs[i].idx2pos(idx_x_max[i])
                if x[i]<xmin:
                    notInRange = True
                    expansion_direction[i] = -1
                if x[i]>xmax:
                    notInRange = True
        return notInRange, expansion_direction
        
class LocalRefinementSimplexTesselationCubicInterpolation(IncrementalLearningSystem):
    """ Layered Simplicial Tesselation with linear interpolation
    
    """
    def __init__(self, foot):
        IncrementalLearningSystem.__init__(self,foot)
        self.delta = foot["delta"]
        try:
            self.initRes = foot["res"]
        except KeyError:
            self.initRes = 1.0
            
        try:
            self.initOffset = foot["offset"]
        except KeyError:
            self.initOffset = 0.0
            
    def prepare(self, antecessor):
        self.nrIn = int(np.size(antecessor["xLearn"].output))
        self.nrLayer = 1
        self.layer = []
        self.layer.append(CubicSimplicialBSplineLayer(self.nrIn, self.initRes, self.initOffset, self.delta))
        
    def evaluate(self, x):
        allPhi = []
        allIdx = []
        allNodes = []
        out = 0.0
        layer_index = 0
        layer_approx = self.layer[layer_index].approximator
        phiX, idxX = layer_approx.aggreg(x)
        nodes = layer_approx.alpha[idxX]
        allPhi.append(phiX)
        allIdx.append(idxX)
        allNodes.append(nodes)
        layer_index += 1
        while layer_index<self.nrLayer and any(nodes):            
            layer_approx = self.layer[layer_index].approximator
            phiX, idxX = layer_approx.aggreg(x)
            nodes = layer_approx.alpha[idxX]
            if any(nodes):
                allPhi.append(phiX)
                allIdx.append(idxX)
                allNodes.append(nodes)
            layer_index += 1
        currentDepth = len(allNodes)-1
        current_nodes = allNodes[currentDepth]
        if current_nodes[0] is None:
            #find base node in upper layer(s)
            print allIdx[-1][0]
            print allIdx[-1][0]/2
            pass
        for i in range(len(current_nodes)):
            if current_nodes[i]:
                out += phiX[i]*current_nodes[i].value
            else:
                pass
            
        return out
        
    def learn(self, x, y):
        notInRange, expansion_direction = self.getExpensionDirection(x)
        while notInRange:
            self.expandRange(expansion_direction)
            notInRange, expansion_direction = self.getExpensionDirection(x)
            
#        print x,y
        allPhi = []
        allIdx = []
        allNodes = []
        layer_index = 0
        layer_approx = self.layer[layer_index].approximator
        phiX, idxX = layer_approx.aggreg(x)
        nodes = layer_approx.alpha[idxX]
        allPhi.append(phiX)
        allIdx.append(idxX)
        allNodes.append(nodes)
        layer_index += 1
        while layer_index<self.nrLayer and any(nodes):            
            layer_approx = self.layer[layer_index].approximator
            phiX, idxX = layer_approx.aggreg(x)
            nodes = layer_approx.alpha[idxX]
            if any(nodes):
                allPhi.append(phiX)
                allIdx.append(idxX)
                allNodes.append(nodes)
            layer_index += 1
        
        out = 0.0
        out_C = 0.0
        out_std = 0.0
        currentDepth = len(allNodes)-1
        current_nodes = allNodes[currentDepth]
        current_phi = allPhi[currentDepth]
        if any(current_nodes):
            current_C = np.zeros(len(current_nodes))
            for i in range(len(current_nodes)):
                if current_nodes[i]:
                    out += current_phi[i]*current_nodes[i].value
                    out_C += current_phi[i]*current_nodes[i].C
                    out_C += current_phi[i]*current_nodes[i].std
                    current_C[i] = current_nodes[i].C
                else:
                    pass
            error = y-out
#            delta = error*current_phi/np.sum(current_phi**2)
#            print x,y,out,delta,allPhi[currentDepth]
            for i in range(len(current_nodes)):
                if current_nodes[i]:
                    blend = current_phi[i]/(current_phi[i]+current_C[i])
                    current_nodes[i].value = current_nodes[i].value*(1-blend) + y*blend
                    current_nodes[i].C += current_phi[i]
                    current_nodes[i].std = current_nodes[i].std*(1-blend) + abs(error)*blend
                else:
                    pass
        else:
#            print allPhi
#            print allIdx
#            print allNodes
#            print y
            idx = allIdx[-1]
            for i in range(len(idx)):
                self.layer[-1].approximator.alpha[idx[i]] = y
                self.layer[-1].approximator.alpha[idx[i],'C'] = current_phi[i]
                self.layer[-1].approximator.alpha[idx[i],'std'] = np.zeros(1)
                
        if self.nrLayer==1:
            no_noise = True
        else:
            no_noise = False
        if no_noise and out_C>(self.nrIn+1)*10:
#            if currentDepth<self.nrLayer-1:
#                self.layer[layer_index].learn(x,y_residual)
#            elif self.nrLayer<10:
#                print layerStd
            next_res = 0.5 * self.layer[-1].res
            next_offset = self.layer[-1].offset
            self.layer.append(CubicSimplicialBSplineLayer(self.nrIn, next_res, next_offset, self.delta))
            layer_approx = self.layer[-1].approximator
            phiX, idxX = layer_approx.aggreg(x)
            nodes = layer_approx.alpha[idxX]
            for i in range(1,len(nodes)):
                layer_approx.alpha[idxX[i]] = y
                layer_approx.alpha[idxX[i],'C'] = phiX[i]
                layer_approx.alpha[idxX[i],'std'] = np.array(out_std)
            self.nrLayer += 1
            print "Adding new layer @res", next_res, " number of layers:",self.nrLayer

    
    def expandRange(self, expansion_direction):
        newOffset = self.layer[0].offset.copy()
        res = self.layer[0].res
        for i in xrange(self.nrIn):
            newOffset[i] += expansion_direction[i]*res[i]
            
        new_top_layer = LinearSimplicialBSplineLayer(self.nrIn, 2*res, newOffset, self.delta)
        
        node_idx = self.layer[0].approximator.alpha.toList('multiDimKey')
        node_C = self.layer[0].approximator.alpha.toList('C')
        for i in xrange(len(node_idx)):
            current_pos = self.layer[0].approximator.inputs.idx2pos(node_idx[i])
            phiX, idxX = new_top_layer.approximator.aggreg(current_pos)
            for j in xrange(len(phiX)):
                new_top_layer.approximator.alpha[idxX[j],'C'] += phiX[j]*node_C[i]
        self.layer.insert(0,new_top_layer)
        self.nrLayer += 1
        print "Expanding Range to res",2*res
        
    def getExpensionDirection(self, x):
        notInRange = False
        idx_x_min = self.layer[0].approximator.coordTransform.minDim
        idx_x_max = self.layer[0].approximator.coordTransform.maxDim
        if not idx_x_min:
            return False, np.zeros(self.nrIn)
        expansion_direction = np.zeros(self.nrIn)
        if self.nrIn==1:
            xmin = self.layer[0].approximator.inputs[0].idx2pos(idx_x_min)
            xmax = self.layer[0].approximator.inputs[0].idx2pos(idx_x_max)
            if x<xmin:
                notInRange = True
#                expansion_direction[0] = -1
            if x>xmax:
                notInRange = True
        else:
            for i in xrange(self.nrIn):
                xmin = self.layer[0].approximator.inputs[i].idx2pos(idx_x_min[i])
                xmax = self.layer[0].approximator.inputs[i].idx2pos(idx_x_max[i])
                if x[i]<xmin:
                    notInRange = True
                    expansion_direction[i] = -1
                if x[i]>xmax:
                    notInRange = True
        return notInRange, expansion_direction
        
class MRSTCubicInterpolation(IncrementalLearningSystem):
    """ Multi Resolution Simplex Tesselation with Cubic Interpolation
    
    """
    def __init__(self, foot):
        IncrementalLearningSystem.__init__(self,foot)
#        self.delta = foot["delta"]
        if not hasattr(self,"res"):
            self.res = 1.0
        if not hasattr(self,"offset"):
            self.offset = 0.0
        if not hasattr(self,"nrLayer"):
            self.nrLayer = 5
            
    def prepare(self, antecessor):
        self.nrIn = int(np.size(antecessor["xLearn"].output))
        self.monomialLayer = []        
        self.monomialLayer.append(MonomialLayer(self.nrIn,0))
        self.monomialLayer.append(MonomialLayer(self.nrIn,1))
        
        self.layer = []
        for i in range(self.nrLayer):
            self.layer.append(CubicSimplicialBSplineLayer(self.nrIn, self.res*2**(-i), self.offset))
        
    def evaluate(self, x):
        out,C = self.monomialLayer[0].evaluateC(x)
        y,c = self.monomialLayer[1].evaluateC(x)
        if c>=4.0:
            out = y
        for i in xrange(self.nrLayer):
            y,c = self.layer[i].evaluateC(x)
            if c>=4.0:
                out = y
            
        return out
        
    def learn(self, x, y):
        notInRange, expansion_direction = self.getExpensionDirection(x)
        while notInRange:
            self.expandRange(expansion_direction)
            notInRange, expansion_direction = self.getExpensionDirection(x)
        
        for i in self.monomialLayer:
            i.learn(x,y)
        for i in self.layer:
            i.learn(x,y)


    
    def expandRange(self, expansion_direction):
        newOffset = self.layer[0].offset.copy()
        res = self.layer[0].res
        for i in xrange(self.nrIn):
            newOffset[i] += expansion_direction[i]*res[i]
            
        new_top_layer = CubicSimplicialBSplineLayer(self.nrIn, 2*res, newOffset)
        
        node_idx = self.layer[0].approximator.alpha.toList('multiDimKey')
        node_C = self.layer[0].approximator.alpha.toList('C')
        for i in xrange(len(node_idx)):
            current_pos = self.layer[0].approximator.inputs.idx2pos(node_idx[i])
            phiX, idxX = new_top_layer.approximator.aggreg(current_pos)
            for j in xrange(len(phiX)):
                new_top_layer.approximator.alpha[idxX[j],'C'] += phiX[j]*node_C[i]
        self.layer.insert(0,new_top_layer)
        self.nrLayer += 1
        print "Expanding Range to res",2*res
        
    def getExpensionDirection(self, x):
        notInRange = False
        idx_x_min = self.layer[0].approximator.coordTransform.minDim
        idx_x_max = self.layer[0].approximator.coordTransform.maxDim
        if idx_x_min is None:
            return False, np.zeros(self.nrIn)
        expansion_direction = np.zeros(self.nrIn)
        if self.nrIn==1:
            xmin = self.layer[0].approximator.inputs[0].idx2pos(idx_x_min)
            xmax = self.layer[0].approximator.inputs[0].idx2pos(idx_x_max)
            if x<xmin:
                notInRange = True
#                expansion_direction[0] = -1
            if x>xmax:
                notInRange = True
        else:
            for i in xrange(self.nrIn):
                xmin = self.layer[0].approximator.inputs[i].idx2pos(idx_x_min[i])
                xmax = self.layer[0].approximator.inputs[i].idx2pos(idx_x_max[i])
                if x[i]<xmin:
                    notInRange = True
                    expansion_direction[i] = -1
                if x[i]>xmax:
                    notInRange = True
        return notInRange, expansion_direction