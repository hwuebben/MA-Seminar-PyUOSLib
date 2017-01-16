# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:57:40 2015

@author: JSCHOENK
"""
from __future__ import division
import numpy as np
from scipy import integrate

class InputContainer:
    """Container for input basis function
    
    """
    def __init__(self,foot):
        self.rawIn = foot["inputs"]
        self.nrIn = len(self.rawIn)
        self.inputs = []
            
        for i in xrange(self.nrIn):
            exec("from BaseFunction import "+ self.rawIn[i]['kind'])
            exec('self.inputs.append(' + self.rawIn[i]['kind'] 
                 + '(self.rawIn[i]))')

    def __call__(self, x):
        out = []
        for i in xrange(self.nrIn):
            out.append(self.inputs[i](x[i]))
        return out
        
    def __getitem__(self, index):
        return self.inputs[index]
            
class SparseInputContainer:
    """Container for sparse input basis function
    
    """
    def __init__(self,foot):
        self.rawIn = foot["inputs"]
        self.nrIn = len(self.rawIn)
        self.inputs = []

        for i in xrange(self.nrIn):
            exec("from BaseFunction import Sparse"+ self.rawIn[i]['kind'])
            exec('self.inputs.append(Sparse' + self.rawIn[i]['kind'] 
                 + '(self.rawIn[i]))')
            
    def __call__(self, x):
        phi = []
        idx = []
        for i in xrange(self.nrIn):
            phiX, idxX = self.inputs[i](x[i])
            phi.append(phiX)
            idx.append(idxX)
        return phi, idx
        
    def idx2pos(self, idx):
        out = np.zeros(self.nrIn)
        if self.nrIn == 1:
            out = self.inputs[0].idx2pos(idx)
        else:
            for i in xrange(self.nrIn):
                out[i] = self.inputs[i].idx2pos(idx[i])
        return out
        
    def __getitem__(self, index):
        return self.inputs[index]

class CoordTransform():
    """Z-Transformation based indexing of multi dimensional coordinate
    
    """
    def __init__(self,minInEachDim,maxInEachDim=None):
        self.nrIn = len(minInEachDim)
        if minInEachDim==None:
            self.minDim = np.zeros(self.nrIn)
            self.maxDim = minInEachDim
        else:
            self.minDim = minInEachDim
            self.maxDim = maxInEachDim
        self.bitsPerInput = np.array(np.ceil(np.log2(self.maxDim-self.minDim+1)),dtype=int)
        
    def __call__(self,multiDim):
        reducedInput = multiDim-self.minDim
        index = 0
        out = 0
        for i in xrange(max(self.bitsPerInput)):
            for j in xrange(self.nrIn):
                if self.bitsPerInput[j]>i:
                    out += (int(reducedInput[j]) & 1<<i)<<(index-i)
                    index += 1        
        return out
        
class AdaptiveCoordTransform():
    """Z-Transformation based adaptive indexing of multi dimensional coordinate
    
    """
    def __init__(self):
        self.minDim = None
        self.maxDim = None
        self.version = 0
        self.bitsPerInput = []
        self.maxbitsPerInput = 0
        
    def updateBitsPerInput(self):
#        print "--- Updating bits per Input - minDim:", self.minDim, "maxDim:", self.maxDim
        self.bitsPerInput = np.array(np.ceil(np.log2(self.maxDim-self.minDim+1)),dtype=int)
        if self.nrIn==1:
            self.maxbitsPerInput = self.bitsPerInput
        else:
            self.maxbitsPerInput = max(self.bitsPerInput)
        self.version += 1
        
    def transform(self, multiDim):
        amultiDim = np.array(multiDim)
        try:
            reducedInput = amultiDim-self.minDim
        except ValueError:
            return None
        except TypeError:
            return None
        if np.size(amultiDim)==1:
            if amultiDim<self.minDim:
                return None
            if amultiDim>self.maxDim:
                return None
        else:
            for i in xrange(len(amultiDim)):
                if amultiDim[i]<self.minDim[i]:
                    return None
                if amultiDim[i]>self.maxDim[i]:
                    return None
        index = 0
        out = 0
        if self.nrIn==1:
            for i in xrange(self.maxbitsPerInput):
                out += (int(reducedInput) & 1<<i)<<(index-i)
                index += 1
        else:
            for i in xrange(self.maxbitsPerInput):
                for j in xrange(self.nrIn):
                    if i < self.bitsPerInput[j]:
                        out += (int(reducedInput[j]) & 1<<i)<<(index-i)
                        index += 1
        return out
        
    def transform_and_add(self,multiDim):      
        amultiDim = np.array(multiDim)
        try:
            reducedInput = amultiDim-self.minDim
        except ValueError:
            self.minDim = amultiDim.copy()
            self.maxDim = amultiDim.copy()
            self.nrIn = np.size(self.minDim)
            self.updateBitsPerInput()
            return 0
        except TypeError:
            self.minDim = amultiDim.copy()
            self.maxDim = amultiDim.copy()
            self.nrIn = np.size(self.minDim)
            self.updateBitsPerInput()
            return 0
        doUpdate = False
        if np.size(amultiDim)==1:
            if amultiDim<self.minDim:
                self.minDim = np.array(amultiDim)
                doUpdate = True
            if amultiDim>self.maxDim:
                self.maxDim = np.array(amultiDim)
                doUpdate = True
        else:
            for i in xrange(len(amultiDim)):
                if amultiDim[i]<self.minDim[i]:
                    self.minDim[i] = amultiDim[i]
                    doUpdate = True
                if amultiDim[i]>self.maxDim[i]:
                    self.maxDim[i] = amultiDim[i]
                    doUpdate = True
        if doUpdate:
            self.updateBitsPerInput()
        index = 0
        out = 0
        if self.nrIn==1:
            for i in xrange(self.maxbitsPerInput):
                out += (int(reducedInput) & 1<<i)<<(index-i)
                index += 1
        else:
            for i in xrange(self.maxbitsPerInput):
                for j in xrange(self.nrIn):
                    if i < self.bitsPerInput[j]:
                        out += (int(reducedInput[j]) & 1<<i)<<(index-i)
                        index += 1
        return out

class AVLNode():
    """ AVL tree node
    
    """
    def __init__(self, key):
        self.key = key
        self.parent = None
        self.leftChild = None
        self.rightChild = None
        self.height = 0 
    
    def __str__(self):
        return str(self.key) + " @(" + str(self.height) + ")"
    
    def is_leaf(self):
        return (self.height == 0)
   
    def max_children_height(self):
        if self.leftChild and self.rightChild:
            return max(self.leftChild.height, self.rightChild.height)
        elif self.leftChild and not self.rightChild:
            return self.leftChild.height
        elif not self.leftChild and  self.rightChild:
            return self.rightChild.height
        else:
            return -1
        
    def balance (self):
        return (self.leftChild.height if self.leftChild else -1) - (self.rightChild.height if self.rightChild else -1)
        
class AVLTree():
    """AVL tree
    
    """
    def __init__(self):
        self.rootNode = None
        self.elements_count = 0
        self.rebalance_count = 0
        
    def height(self):
        if self.rootNode:
            return self.rootNode.height
        else:
            return 0
        
    def rebalance (self, node_to_rebalance):
        self.rebalance_count += 1
        A = node_to_rebalance 
        F = A.parent #allowed to be NULL
        if node_to_rebalance.balance() == -2:
            if node_to_rebalance.rightChild.balance() <= 0:
                """Rebalance, case RRC """
                B = A.rightChild
                C = B.rightChild
                assert (not A is None and not B is None and not C is None)
                A.rightChild = B.leftChild
                if A.rightChild:
                    A.rightChild.parent = A
                B.leftChild = A
                A.parent = B                                                               
                if F is None:                                                              
                   self.rootNode = B 
                   self.rootNode.parent = None                                                   
                else:                                                                        
                   if F.rightChild == A:                                                          
                       F.rightChild = B                                                                  
                   else:                                                                      
                       F.leftChild = B                                                                   
                   B.parent = F 
                self.recompute_heights(A) 
                self.recompute_heights(B.parent)                                                                                         
            else:
                """Rebalance, case RLC """
                B = A.rightChild
                C = B.leftChild
                assert (not A is None and not B is None and not C is None)
                B.leftChild = C.rightChild
                if B.leftChild:
                    B.leftChild.parent = B
                A.rightChild = C.leftChild
                if A.rightChild:
                    A.rightChild.parent = A
                C.rightChild = B
                B.parent = C                                                               
                C.leftChild = A
                A.parent = C                                                             
                if F is None:                                                             
                    self.rootNode = C
                    self.rootNode.parent = None                                                    
                else:                                                                        
                    if F.rightChild == A:                                                         
                        F.rightChild = C                                                                                     
                    else:                                                                      
                        F.leftChild = C
                    C.parent = F
                self.recompute_heights(A)
                self.recompute_heights(B)
        else:
            assert(node_to_rebalance.balance() == +2)
            if node_to_rebalance.leftChild.balance() >= 0:
                B = A.leftChild
                C = B.leftChild
                """Rebalance, case LLC """
                assert (not A is None and not B is None and not C is None)
                A.leftChild = B.rightChild
                if (A.leftChild): 
                    A.leftChild.parent = A
                B.rightChild = A
                A.parent = B
                if F is None:
                    self.rootNode = B
                    self.rootNode.parent = None                    
                else:
                    if F.rightChild == A:
                        F.rightChild = B
                    else:
                        F.leftChild = B
                    B.parent = F
                self.recompute_heights(A)
                self.recompute_heights(B.parent) 
            else:
                B = A.leftChild
                C = B.rightChild 
                """Rebalance, case LRC """
                assert (not A is None and not B is None and not C is None)
                A.leftChild = C.rightChild
                if A.leftChild:
                    A.leftChild.parent = A
                B.rightChild = C.leftChild
                if B.rightChild:
                    B.rightChild.parent = B
                C.leftChild = B
                B.parent = C
                C.rightChild = A
                A.parent = C
                if F is None:
                   self.rootNode = C
                   self.rootNode.parent = None
                else:
                   if (F.rightChild == A):
                       F.rightChild = C
                   else:
                       F.leftChild = C
                   C.parent = F
                self.recompute_heights(A)
                self.recompute_heights(B)
                    
    def recompute_heights (self, start_from_node):
        changed = True
        node = start_from_node
        while node and changed:
            old_height = node.height
            node.height = (node.max_children_height() + 1 if (node.rightChild or node.leftChild) else 0)
            changed = node.height != old_height
            node = node.parent
       
    def add_as_child (self, parent_node, child_node):
        node_to_rebalance = None
        if child_node.key < parent_node.key:
            if not parent_node.leftChild:
                parent_node.leftChild = child_node
                child_node.parent = parent_node
                if parent_node.height == 0:
                    node = parent_node
                    while node:
                        node.height = node.max_children_height() + 1
                        if not node.balance () in [-1, 0, 1]:
                            node_to_rebalance = node
                            break #we need the one that is furthest from the root
                        node = node.parent     
            else:
                self.add_as_child(parent_node.leftChild, child_node)
        else:
            if not parent_node.rightChild:
                parent_node.rightChild = child_node
                child_node.parent = parent_node
                if parent_node.height == 0:
                    node = parent_node
                    while node:
                        node.height = node.max_children_height() + 1
                        if not node.balance () in [-1, 0, 1]:
                            node_to_rebalance = node
                            break #we need the one that is furthest from the root
                        node = node.parent       
            else:
                self.add_as_child(parent_node.rightChild, child_node)
        
        if node_to_rebalance:
            self.rebalance (node_to_rebalance)
                
    def insert (self, key):
        new_node = AVLNode(key)
        if not self.rootNode:
            self.rootNode = new_node
        else:
            if not self.find(key):
                self.elements_count += 1
                self.add_as_child (self.rootNode, new_node)
                          
    def find(self, key):
        return self.find_in_subtree (self.rootNode, key )
    
    def find_in_subtree (self,  node, key):
        if node is None:
            return None  # key not found
        if key < node.key:
            return self.find_in_subtree(node.leftChild, key)
        elif key > node.key:
            return self.find_in_subtree(node.rightChild, key)
        else:  # key is equal to node key
            return node
            
    def remove (self, key):
        # first find
        node = self.find(key)
        
        if not node is None:
            self.elements_count -= 1
            
            #     There are three cases:
            # 
            #     1) The node is a leaf.  Remove it and return.
            # 
            #     2) The node is a branch (has only 1 child). Make the pointer to this node 
            #        point to the child of this node.
            # 
            #     3) The node has two children. Swap items with the successor
            #        of the node (the smallest item in its right subtree) and
            #        delete the successor from the right subtree of the node.
            if node.is_leaf():
                self.remove_leaf(node)
            elif (bool(node.leftChild)) ^ (bool(node.rightChild)):  
                self.remove_branch (node)
            else:
                assert (node.leftChild) and (node.rightChild)
                self.swap_with_successor_and_remove (node)
                
    def remove_leaf (self, node):
        parent = node.parent
        if (parent):
            if parent.leftChild == node:
                parent.leftChild = None
            else:
                assert (parent.rightChild == node)
                parent.rightChild = None
            self.recompute_heights(parent)
        else:
            self.rootNode = None
        del node
        # rebalance
        node = parent
        while (node):
            if not node.balance() in [-1, 0, 1]:
                self.rebalance(node)
            node = node.parent
            
    def remove_branch (self, node):
        parent = node.parent
        if (parent):
            if parent.leftChild == node:
                parent.leftChild = node.rightChild or node.leftChild
            else:
                assert (parent.rightChild == node)
                parent.rightChild = node.rightChild or node.leftChild
            if node.leftChild:
                node.leftChild.parent = parent
            else:
                assert (node.rightChild)
                node.rightChild.parent = parent 
            self.recompute_heights(parent)
        del node
        # rebalance
        node = parent
        while (node):
            if not node.balance() in [-1, 0, 1]:
                self.rebalance(node)
            node = node.parent
            
    def swap_with_successor_and_remove (self, node):
        successor = self.find_smallest(node.rightChild)
        self.swap_nodes (node, successor)
        assert (node.leftChild is None)
        if node.height == 0:
            self.remove_leaf (node)
        else:
            self.remove_branch (node)
            
    def swap_nodes (self, node1, node2):
        assert (node1.height > node2.height)
        parent1 = node1.parent
        leftChild1 = node1.leftChild
        rightChild1 = node1.rightChild
        parent2 = node2.parent
        assert (not parent2 is None)
        assert (parent2.leftChild == node2 or parent2 == node1)
        leftChild2 = node2.leftChild
        assert (leftChild2 is None)
        rightChild2 = node2.rightChild
        
        # swap heights
        tmp = node1.height 
        node1.height = node2.height
        node2.height = tmp
       
        if parent1:
            if parent1.leftChild == node1:
                parent1.leftChild = node2
            else:
                assert (parent1.rightChild == node1)
                parent1.rightChild = node2
            node2.parent = parent1
        else:
            self.rootNode = node2
            node2.parent = None
            
        node2.leftChild = leftChild1
        leftChild1.parent = node2
        node1.leftChild = leftChild2 # None
        node1.rightChild = rightChild2
        if rightChild2:
            rightChild2.parent = node1 
        if not (parent2 == node1):
            node2.rightChild = rightChild1
            rightChild1.parent = node2
            
            parent2.leftChild = node1
            node1.parent = parent2
        else:
            node2.rightChild = node1
            node1.parent = node2
            
class ParameterNode(AVLNode):
    """Parameter tree node
    
    """
    def __init__(self, key, value):
        AVLNode.__init__(self,key)
        self.value = value
    
    def __str__(self):
        return str(self.key) + ":" + str(self.value) + " @(" + str(self.height) + ")"
        
        
class AdaptiveParameterNode(AVLNode):
    """Parameter tree node
    
    """
    def __init__(self, key, coordTransform):
        AVLNode.__init__(self,key)
        self.coordTransform = coordTransform
        self.version = np.array(coordTransform.version)
        self.multiDimKey = np.array(key)
        self.key = np.array(self.coordTransform.transform_and_add(self.multiDimKey))
        
    def get_key(self):
        if self.version<self.coordTransform.version:
            self.key = np.array(self.coordTransform.transform(self.multiDimKey))
            self.version = self.coordTransform.version
        return self.key
            
    def __str__(self):
        return str(self.key) + ":" + str(self.value) + " @(" + str(self.height) + ")"
        
    def toList(self, out, attr = 'value'):        
        if self.leftChild:
            self.leftChild.toList(out, attr)
        out.append(getattr(self,attr))
        if self.rightChild:
            self.rightChild.toList(out, attr)
            
class ParameterTree(AVLTree):
    def __init__(self, defaultValue = 0.0):
        AVLTree.__init__(self)
        self.defVal = defaultValue
        
            
    def __setitem__(self,key,value):
        try:
            keyLen = len(key)
        except TypeError:
            keyLen = 1
        if keyLen == 1:
            tmp = self.find(np.array(key))
            if tmp==None:
                self.insert(key,value)
            else:
                tmp.value = value
        else:
            if np.shape(key)==np.shape(value):
                for i in xrange(keyLen):
                    tmp = self.find(key[i])
                    if tmp == None:
                        self.insert(key[i],value[i])
                    else:
                        tmp.value = value[i]
    
    def insert (self, key, value):
        new_node = ParameterNode(np.array(key), value)
        if not self.rootNode:
            self.rootNode = new_node
        else:
            if not self.find(key):
                self.elements_count += 1
                self.add_as_child (self.rootNode, new_node)
                
    def __getitem__(self,key):
        try:
            keyLen = len(key)
        except TypeError:
            keyLen = 1
        if keyLen == 1:
            tmp = self.find(key)
            if tmp==None:
                return np.array(self.defVal)
            else:
                return np.array(tmp.value)
        else:
            out = np.zeros(keyLen)
            for i in xrange(keyLen):
                tmp = self.find(key[i])
                if tmp==None:
                    out[i] = self.defVal
                else:
                    out[i] = tmp.value
            return out

class AdaptiveParameterTree(AVLTree):
    def __init__(self, coordTransform, defaultValue = None):
        AVLTree.__init__(self)
        self.coordTransform = coordTransform
        self.defaults = {"value":defaultValue}
       
    def add_as_child (self, parent_node, child_node):
        node_to_rebalance = None
        if child_node.get_key() < parent_node.get_key():
            if not parent_node.leftChild:
                parent_node.leftChild = child_node
                child_node.parent = parent_node
                if parent_node.height == 0:
                    node = parent_node
                    while node:
                        node.height = node.max_children_height() + 1
                        if not node.balance () in [-1, 0, 1]:
                            node_to_rebalance = node
                            break #we need the one that is furthest from the root
                        node = node.parent     
            else:
                self.add_as_child(parent_node.leftChild, child_node)
        else:
            if not parent_node.rightChild:
                parent_node.rightChild = child_node
                child_node.parent = parent_node
                if parent_node.height == 0:
                    node = parent_node
                    while node:
                        node.height = node.max_children_height() + 1
                        if not node.balance () in [-1, 0, 1]:
                            node_to_rebalance = node
                            break #we need the one that is furthest from the root
                        node = node.parent       
            else:
                self.add_as_child(parent_node.rightChild, child_node)
        
        if node_to_rebalance:
            self.rebalance (node_to_rebalance)
                
    def insert (self, key, value = None, attribute = 'value'):
        new_node = AdaptiveParameterNode(key, self.coordTransform)
        for attribute,value in self.defaults.iteritems():
            try:
                setattr(new_node,attribute,value.copy())
            except AttributeError:
                setattr(new_node,attribute,value)
        if not value:
            try:
                setattr(new_node,attribute,value.copy())
            except AttributeError:
                setattr(new_node,attribute,value)
            
        if not self.rootNode:
            self.rootNode = new_node
            return new_node
        else:
            if not self.find(key):
                self.elements_count += 1
                self.add_as_child(self.rootNode, new_node)
                return new_node
            else:
                return None
    def find(self, key):
        one_dim_key = self.coordTransform.transform(key)
        return self.find_in_subtree (self.rootNode, one_dim_key )
    
    def find_in_subtree (self,  node, one_dim_key):
        if node is None:
            return None  # key not found
        if one_dim_key < node.get_key():
            return self.find_in_subtree(node.leftChild, one_dim_key)
        elif one_dim_key > node.get_key():
            return self.find_in_subtree(node.rightChild, one_dim_key)
        else:  # key is equal to node key
            return node
    
    def __getitem__(self, key_in):
        if isinstance(key_in,tuple):
            if len(key_in)==2:
                key = key_in[0]
                attribute = key_in[1]
            else:
                print "Error: don't know how to hanlde this tuple", key
        else:
            key = key_in
            attribute = None
        if len(key) == 1:
            if attribute==None:
                out = [self.find(key)]
            else:
                node = self.find(key)
                if node is None:
                    out = [self.defaults[attribute]]
                else:
                    try:
                        out = [getattr(node,attribute)]
                    except AttributeError:
                        out = [self.defaults[attribute]]
        else:
            nrOut = np.shape(key)[0]
            out = []
            for i in xrange(nrOut):
                if attribute==None:
                    out.append(self.find(key[i]))
                else:
                    node = self.find(key[i])
                    if node == None:
                        out.append(self.defaults[attribute])
                    else:
                        out.append(getattr(node,attribute))
        if not attribute:
            return out
        else:
            return np.array(out).flatten()
            
    def __setitem__(self, key_in, value_in):
        value = value_in.copy()
        if isinstance(key_in,tuple):
            if len(key_in)==2:
                key = key_in[0]
                attribute = key_in[1]
            else:
                print "Error: don't know how to hanlde this tuple",key
        else:
            key = key_in
            attribute = 'value'
            
        if np.size(key) == 1:
            tmp = self.find(key)
            if tmp==None:
                self.insert(key, value, attribute)
            else:
                tmp.value = value
        elif len(np.shape(key))==1:
            tmp = self.find(key)
            if tmp==None:
                self.insert(key, value, attribute)
            else:
                tmp.value = value
        else:
            if np.shape(key)[0]==len(value):
                for i in xrange(len(value)):
                    tmp = self.find(key[i])
                    if tmp == None:
                        self.insert(key[i],value[i], attribute)
                    else:
                        setattr(tmp, attribute, value[i])
                        
    def toList(self, attr = 'value'):
        nodes = []
        self.rootNode.toList(nodes, attr)
        return np.array(nodes)
        

class ParameterContainer:
    def __init__(self,foot,nrAlpha):
        try:
            self.alpha0 = np.array(foot["alpha0"])
        except KeyError:
            self.alpha0 = np.zeros(1)
            
        self.nrAlpha = nrAlpha            
        if len(self.alpha0) == self.nrAlpha:
            self.alpha = self.alpha0
        elif len(self.alpha0) == 1:
            self.alpha = np.zeros(self.nrAlpha)+self.alpha0
        else:
            print "WARNING: INVALID INITIALIZATION OF PARAMETER ALPHA, \
            SETING ALPHA TO ZERO"
            self.alpha = np.zeros(self.nrAlpha)
            
    def reset(self):
        lenAlpha0 = len(self.alpha0)
        if lenAlpha0 == self.nrAlpha:
            self.alpha = self.alpha0
        elif lenAlpha0 == 1:
            self.alpha = np.zeros(self.nrAlpha)+self.alpha0
        else:
            print "WARNING: INVALID INITIALIZATION OF PARAMETER ALPHA, \
            SETING ALPHA TO ZERO"
            self.alpha = np.zeros(self.nrAlpha)
            
class SparseParameterContainer:
    """Sparse Parameter Container
    
    """
    def __init__(self,foot):
        try:
            self.alpha0 = np.array(foot["alpha0"])
        except KeyError:
            self.alpha0 = np.zeros(1)
            
        self.alpha = ParameterTree(self.alpha0)
            
    def reset(self):
        self.alpha = ParameterTree(self.alpha0)
        
class TreeParameterContainer:
    """Tree-based adaptive parameter container
    
    """
    def __init__(self,foot):
        try:
            self.defaultValue = np.array(foot["defaultValue"])
            
        except KeyError:
            self.defaultValue = np.zeros(1)
        self.coordTransform = AdaptiveCoordTransform()
        self.alpha = AdaptiveParameterTree(self.coordTransform,self.defaultValue)

    def reset(self):
        self.coordTransform = AdaptiveCoordTransform()
        self.alpha = AdaptiveParameterTree(self.coordTransform,self.defaultValue)
            
            
class LIPApproximator:
    """Common interface for LIP-Approximators
    
    """
    def __init__(self):
        pass
            
    def evaluatePhiX(self, phiX):
        return np.dot(phiX, self.alpha)
        
    def __call__(self, x):
        phiX = self.aggreg(x)
        return np.dot(phiX, self.alpha)
        
    def evalD(self, x):
        phiXd = self.aggregD(x)
        out = np.zeros(np.shape(x))
        for i in range(len(out)):
            out[i] = np.inner(phiXd[i],self.alpha)
        return out
        
class SparseLIPApproximator:
    """Common interface for semi sparse LIP-Approximators
    
    """
    def __init__(self):
        pass
            
    def evaluatePhiX(self, phiX, idxX):        
        return np.dot(phiX, self.alpha[idxX])
        
    def __call__(self, x):
        phiX, idxX = self.aggreg(x)
        return np.dot(phiX, self.alpha[idxX])
        
class TreeLIPApproximator:
    """Common interface for sparse LIP-Approximators
    
    """
    def __init__(self):
        pass
            
    def evaluatePhiX(self, phiX, idxX):     
        return np.inner(phiX, self.alpha[idxX,'value'])
        
    def __call__(self, x):
        phiX, idxX = self.aggreg(x)
        return np.inner(phiX, self.alpha[idxX,'value'])
        
class LinearSimplicialBSplineLayer:
    """Adaptive Linear Simplicial B-Spline Learning System
    
    """
    def __init__(self, nrIn, res = 1.0, offset = 0.0, delta = 0.1):
        from Approximator import AdaptiveLinearSimplicialBSpline
        from Learner import TreevSGD
        self.nrIn = nrIn
        self.delta = delta
        approx_input = {"inputs":[]}
        if np.size(res) == 1:
            single_res = np.zeros(self.nrIn)+res
        elif np.size(res) == self.nrIn:
            single_res = np.array(res)
        else:
            print "Error: size(res) is not 1 and not nrIn:", self.nrIn
        self.res = single_res
        if np.size(offset) == 1:
            single_offset = np.zeros(self.nrIn)+offset
        elif np.size(offset) == self.nrIn:
            single_offset = np.array(res)
        else:
            print "Error: size(offset) is not 1 and not nrIn:", self.nrIn
        self.offset = single_offset
        for i in xrange(self.nrIn):
            approx_input["inputs"].append({"kind":"UIGLTlinear","res":single_res[i],"offset":single_offset[i]})
            
                
        self.approximator = AdaptiveLinearSimplicialBSpline(approx_input)
        
        
        # setup learner
        self.learner = TreevSGD(self.approximator,{})
        
        
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
        sumC = 0.0
        for i in xrange(len(nodes)):
            nodes[i].value += deltaAlpha[i]
            sumC += nodes[i].C
        return yp,sumC
        
    def learn_and_eval(self, x, y, rate = 1.0):
        phiX, idxX = self.approximator.aggreg(x)
        nodes = self.approximator.alpha[idxX]
        for i in xrange(len(nodes)):
            if nodes[i] == None:
                nodes[i] = self.approximator.alpha.insert(idxX[i])
        yp = self.approximator.evaluatePhiX(phiX,idxX)
        deltaAlpha = self.learner.learn(x, phiX, nodes, y, yp, rate)
        yp += np.inner(phiX,deltaAlpha)
        evalC = 0.0
        evalStd = 0.0
        for i in xrange(len(nodes)):
            nodes[i].value += deltaAlpha[i]
            evalC += nodes[i].C*phiX[i]
            evalStd += nodes[i].std*phiX[i]
        return yp, evalC, evalStd
        
    def vanish(self, x):
        pass
#        phiX, idxX = self.approximator.aggreg(x)
#        nodes = self.approximator.alpha[idxX]
#        for i in xrange(len(nodes)):
#            if not nodes[i] == None:
#                nodes[i].value *= 0.9
#                nodes[i].C -= phiX[i]
#                if nodes[i].C < 0:
#                    nodes[i].C = 0.0
        
    def reset(self):
        self.approximator.reset()
        self.learner.reset()


class CubicSimplicialBSplineLayer:
    """Adaptive Linear Simplicial B-Spline Learning System
    
    """
    def __init__(self, nrIn, res = 1.0, offset = 0.0):
        from Approximator import AdaptiveCubicSimplicialBSpline
        from Learner import TreevSGD
        self.nrIn = nrIn
#        self.delta = delta
        approx_input = {"inputs":[]}
        if np.size(res) == 1:
            single_res = np.zeros(self.nrIn)+res
        elif np.size(res) == self.nrIn:
            single_res = np.array(res)
        else:
            print "Error: size(res) is not 1 and not nrIn:", self.nrIn
        self.res = single_res
        if np.size(offset) == 1:
            single_offset = np.zeros(self.nrIn)+offset
        elif np.size(offset) == self.nrIn:
            single_offset = np.array(res)
        else:
            print "Error: size(offset) is not 1 and not nrIn:", self.nrIn
        self.offset = single_offset
        for i in xrange(self.nrIn):
            approx_input["inputs"].append({"kind":"UIGLTlinear","res":single_res[i],"offset":single_offset[i]})
            
                
        self.approximator = AdaptiveCubicSimplicialBSpline(approx_input)
        
        
        # setup learner
        self.learner = TreevSGD(self.approximator,{})
        
        
    def evaluate(self, x):
        # Evaluation of approximator
        #
        return self.approximator(x)
        
    def evaluateC(self, x):
        phiX, idxX = self.approximator.aggreg(x)
        yp = self.approximator.evaluatePhiX(phiX,idxX)
        evalC = 0.0
        nodes = self.approximator.alpha[idxX]
        for i in xrange(len(nodes)):
            if not nodes[i] == None:
                evalC += nodes[i].C*phiX[i]
        return yp,evalC
                
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
        sumC = 0.0
        for i in xrange(len(nodes)):
            nodes[i].value += deltaAlpha[i]
            sumC += nodes[i].C
        return yp,sumC
        
    def learn_and_eval(self, x, y, rate = 1.0):
        phiX, idxX = self.approximator.aggreg(x)
        nodes = self.approximator.alpha[idxX]
        for i in xrange(len(nodes)):
            if nodes[i] == None:
                nodes[i] = self.approximator.alpha.insert(idxX[i])
        yp = self.approximator.evaluatePhiX(phiX,idxX)
        deltaAlpha = self.learner.learn(x, phiX, nodes, y, yp, rate)
        yp += np.inner(phiX,deltaAlpha)
        evalC = 0.0
        evalStd = 0.0
        for i in xrange(len(nodes)):
            nodes[i].value += deltaAlpha[i]
            evalC += nodes[i].C*phiX[i]
            evalStd += nodes[i].std*phiX[i]
        return yp, evalC, evalStd
        
    def vanish(self, x):
        pass

        
    def reset(self):
        self.approximator.reset()
        self.learner.reset()
        
class MonomialLayer:
    """Adaptive Linear Simplicial B-Spline Learning System
    
    """
    def __init__(self, nrIn, degree):
        self.alpha = np.zeros(1+degree*nrIn)
        self.nrAlpha = len(self.alpha)
        self.degree = degree
        self.C = np.zeros(self.nrAlpha)
        self.std = 0.0
        self.N = 0.0
        
    def phi(self, x):
        out = [1.0]
        for i in xrange(self.degree):
            out.append(x**(i+1))
        out = np.hstack(out)
        return out
        
        
    def evaluate(self, x):
        # Evaluation of approximator
        #
        return np.inner(self.phi(x),self.alpha)
        
    def evaluateC(self, x):
        # Evaluation of approximator
        #
        return np.inner(self.phi(x),self.alpha), self.N/self.nrAlpha
                
    def learn(self, x, y):
        phiX = self.phi(x)
        yp = np.inner(phiX,self.alpha)

        error = y-yp
        grad = phiX*error/np.sum(phiX**2)
        self.alpha += phiX/(self.C+phiX)*grad
        self.C += phiX
        self.std += abs(error)/self.N
        self.N += 1.0
        
    def eval_and_learn(self, x, y):
        phiX = self.phi(x)
        yp = self.inner(phiX,self.alpha)

        error = y-yp
        grad = phiX*error/np.sum(phiX**2)
        self.alpha += phiX/(self.C+phiX)*grad
        sumC = self.C.sum()
        self.C += phiX
        self.std += abs(error)/self.N
        self.N += 1.0
        return yp,sumC
        
    def vanish(self, x):
        pass

        
    def reset(self):
        self.approximator.reset()
        self.learner.reset()