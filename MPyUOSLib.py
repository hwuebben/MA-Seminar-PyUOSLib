# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:57:40 2015

@author: JSCHOENK
"""

from __future__ import division
from time import *

import numpy as np
import json
    
def ModuleImport(module):
    """Helper function to unify the module import
    
    Modules can be imported in different ways and from different directories.
    They may belong to a collection of Modules grouped in one file or have 
    their definition in an own file. These files may be ordered in directories.
    This function checks all key words used in MPyUOSLib in order to 
    correctly import the modules defined in the experiment description.
    """
    tmp = []
    # First check the existance of file reference
    if module.has_key("fileReference"):
        filename = module["fileReference"]
        try:
            with open(filename, "a+") as configFile:
                config = json.load(configFile)
                for key in module.keys():
                    if key is not "fileReference":
                        config[key] = module[key]
                tmp = ModuleImport(config)
        except IOError as exc:
            print "Recursive module file description detected."
            raise exc
    # Second check for nested modular experiments
    elif module["module"] == "ModularExperiment":
        from MPyUOSLib import ModularExperiment
        tmp = ModularExperiment(module, False)
    # Third check for group definitions
    elif module["module"] == "Group":
        from MPyUOSLib import Group
        tmp = Group(module)
    # At this point we are sure to handle a modul
    # but its definition may be located in a subfolder defined as a path
    elif module.has_key("path"):
        path = module["path"]
        # The module may belong to a collection of modules
        if module.has_key("collection"):
            try:
                exec("from Modules." + path + " import " + module["collection"])
                exec("tmp = " + module["collection"] + "." + module["module"] + "(module)")
            except ImportError:
                print "Unable to import module: ", module["module"], " from collection: ", module["collection"], " from path: ", path, ". Check __init__.py file"
                raise
                
        else: # or may have its own file
            try:
                exec("from Modules." + path + " import " + module["module"])
                exec("tmp = " + module["module"] + "." + module["module"] + "(module)")
            except ImportError:
                print "Unable to import module: ", module["module"], " from collection: ", module["collection"], " from path: ", path, ". Check __init__.py file"
                raise
            
    else: # here the module is not located in a subdirectory
        # but still may be part of a collection
        if module.has_key("collection"):
            exec("from Modules. " + module["collection"] + " import " + module["module"])
            exec("tmp = " + module["module"] + "(module)")
        else: # or may have its own file
            exec("from Modules. " + module["module"] + " import " + module["module"])
            exec("tmp = " + module["module"] + "(module)")
    return tmp
    
class BasicProcessingModule:
    """Provides basic features for preparing and running modular experiments
    
    Provides to most common features for data-path orientated experiment
    design and evaluation. All modules within a processing change must 
    inherit this interface. 
    
    The following methods allow a unified handling within the MPyUOSLib 
    framework and must not be changed:
    - connectInputs(self,names)
    - run(self,index)
    
    The other methods allow for a user-specific behaviour of the module:
    - __init__(self,foot)
    - prepare(self,antecessor)
    - __call__(self,argIn,index)
    - end(self)
    - reset(self)
    """
    def __init__(self, foot, default = {}):
        """Standard constructor for processing modules
        
        Saves the footprint and names the modules in order to allow connecting
        inputs.
        
        Inputs:
        foot - Dictionary determining the module type and behaviour
        """
        self.foot = foot                   # save footprint
        self.output = np.zeros(1)          # set default output        
        self.ID = foot["ID"]               # set module ID
        self.parameter = foot["parameter"] # set parameter
        
        # Add all footprint items as attributes to this object
        for k, v in foot.iteritems():
            try:
                # If the value of an attribute points to a global parameter
                if "@" in v:
                    # try to get its value
                    try:
                        setattr(self,k,getattr(self.parameter,v[1:]))
                    except AttributeError: # or print an error
                        print "Error@ID:", self.ID, "(", foot["module"], ") -> Unknown parameter name: ", v[1:], "   Knowns names are:", self.parameter.__dict__.keys()
                else: # There is no pointer to a global parameter
                    setattr(self, k, v)
            except TypeError: # or v is not iterable and thus no string at all
                setattr(self, k, v)
        for k, v in default.iteritems():
            if not hasattr(self, k):
                setattr(self, k, v)
        
    def connectInputs(self, IDs):
        """Connecting moduel inputs
        
        Here all user-defined connections to other modules are established.
        Connections to modules prior in the processing chain provide their 
        current output. Connections to modules subsequent in the processing 
        chain provide their former cycle output.
        
        All inputs are gathered in one dictionary and named according to the
        user-defined descriptors in the module specification.
        
        Inputs:
        names - Dictionary m
        """
        self.inputArguments = {}
        if self.foot.has_key("input"):
            for inputName, moduleID in self.foot["input"].iteritems():
                try:
                    self.inputArguments[inputName] = IDs[moduleID]
                except KeyError:
                    print "Error@ID:", self.ID, "-> Connection failed:", moduleID, "<->", inputName
            
    def prepare(self, antecessor):
        """Allows preparation of the module after connecting its input and 
        before running the experiment
        
        """
        pass
        
    def run(self, index):
        """Entry for ModularExperiments to call a module
        
        Entry for regular module evaluation used by the MPyUOSLib. Here all
        outputs of all defined input modules are gathered in one dictionary in 
        order to call the __call__ method specific to the inheriting module.
        """
        argIn = {"index": index}
        for inputName,moduleReference in self.inputArguments.iteritems():
            argIn[inputName] = moduleReference.output
        self.output = self(**argIn)
        
    def end(self):
        """User-defined function called after running the experiment.
        
        Here one can store values gathered during the experiment, make plots
        and so on.
        """
        pass
    
    def reset(self):
        """User-defined function called to reset the experiment in order to prepare reevaluation
        
        """
        self.output = np.zeros(1)
        
class InputModule(BasicProcessingModule):
    """Pseudo Processing Module supporting Groups and ModularExperiments
    
    """
    def __init__(self, foot):
        """Empty init
        
        """
        foot["parameter"] = {}
        BasicProcessingModule.__init__(self, foot)
        
class TriggerModule(BasicProcessingModule):
    """Basic Trigger Module
    
    
    """
    def __init__(self, foot):
        BasicProcessingModule.__init__(self, foot)
        delattr(self,"output")
        
    def connectInputs(self, IDs):
        """Connecting moduel triggers
        
        Inputs:
        IDs - Dictionary IDs
        """
        self.target = {}
        if self.foot.has_key("target"):
            for targetID,targetFnc in self.foot["target"].iteritems():
                try:
                    self.target[targetID] = getattr(IDs[targetID],targetFnc)
                except KeyError:
                    print "Error@ID:", self.ID, "-> Trigger Connection failed, no module named:", targetID
                except AttributeError:
                    print "Error@ID:", self.ID, "-> Trigger Connection failed, no function named:", targetFnc
                    
        self.inputArguments = {}
        if self.foot.has_key("input"):
            for inputName,moduleID in self.foot["input"].iteritems():
                try:
                    self.inputArguments[inputName] = IDs[moduleID]
                except KeyError:
                    print "Error@ID:", self.ID, "-> Connection failed:", moduleID, "<->",inputName
                    
    def run(self, index):
        """Entry for ModularExperiments to call a module
        
        Entry for regular module evaluation used by the MPyUOSLib. Here all
        outputs of all defined input modules are gathered in one dictionary in 
        order to call the __call__ method specific to the inheriting module.
        """
        argIn = {"index": index}
        for inputName,moduleReference in self.inputArguments.iteritems():
            argIn[inputName] = moduleReference.output
        self(**argIn)
                    
    def reset(self):
        pass
        
class Group(BasicProcessingModule):
    """
    Group of Processing Modules
    
    """
    def __init__(self, foot):
        """Setup and configure all modules defined in this group
        
        """
        BasicProcessingModule.__init__(self,foot)
        
        self.modules = []
        self.moduleIDs = {}
        self.parameter = foot["parameter"]
        moduleTag = foot["modules"]
        self.positionalMode = False
        if type(moduleTag) is list:
            moduleDescription = moduleTag
            self.currentFile = None
            self.fileInputList = None
        else:
            with open(moduleTag) as configFile:
                loadDesc = json.load(configFile)
                if type(loadDesc) is list:
                    moduleDescription = loadDesc
                elif type(loadDesc) is dict:
                    moduleDescription = loadDesc["modules"]
                    self.fileInputList = loadDesc["inputlist"]
                    self.positionalMode = True
                    if not hasattr(self,"inputlist"):
                        self.inputList = self.foot["input"].keys()
                        self.inputList.sort()
                        if "modules" in self.inputList:
                            self.inputList.remove("modules")
            self.currentFile = moduleTag
            
            
        for module in moduleDescription:
            module["parameter"] = self.parameter
            tmp = ModuleImport(module)
            if hasattr(tmp,"nrD"):
                self.nrD = tmp.nrD
            if hasattr(tmp,"simulationSteps"):
                self.simulationSteps = tmp.simulationSteps
            self.modules.append(tmp)
            self.moduleIDs[tmp.ID] = tmp
            
            
        
    def connectInputs(self, IDs):
        """Connect input to the group and all inputs of modules within
        
        Inputs:
        names - Dictionary m
        """
        # Define group input arguments
        self.inputArguments = {}
        if self.foot.has_key("input"):
            for inputName,moduleID in self.foot["input"].iteritems():
                try:
                    self.inputArguments[inputName] = IDs[moduleID]
                except KeyError:
                    print "Error@ID:", self.ID, "-> Connection failed:", moduleID, "<->",inputName
            
        # Rename group input modules and add them to group internal module list
        if self.positionalMode:
            for inputName in self.fileInputList:
                self.moduleIDs[inputName] = InputModule({"ID":inputName})
        else:
            for inputName in self.inputArguments.keys():
                self.moduleIDs[inputName] = InputModule({"ID":inputName})
            
        # Connect inputs of all group internal modules based on extendet
        # internal module list
        for i in self.modules:
            i.connectInputs(self.moduleIDs)

    def prepare(self, antecessor):
        """Prepare all modules within this group
        
        """
        if antecessor.has_key("modules"):
            filename = antecessor["modules"].output
            if filename is not self.currentFile:
                self.reinitModules(filename)
                self.currentFile = filename
        for i in self.modules:
            try:
                i.prepare(i.inputArguments)
            except AttributeError:
                pass #Modules without inputs may decline declaring self.inputArguments
        
    def run(self, index):
        """Collect the input arguments to call this group
        
        """
        argIn = {"index":index}
        if self.positionalMode:
            for i in range(len(self.inputList)):
                argIn[self.fileInputList[i]] = self.inputArguments[self.inputList[i]].output
        else:
            for inputName,moduleReference in self.inputArguments.iteritems():
                argIn[inputName] = moduleReference.output
        self.output = self(**argIn)
        
    def __call__(self, index=0, modules=None, **argIn):
        """Rename the inputs via the input modules for the modules within this group and call them
        
        """                    
        for inputName,moduleOutput in argIn.iteritems():
            self.moduleIDs[inputName].output = moduleOutput
            
        for i in self.modules:
            i.run(index)
            
        return self.modules[-1].output
        
    def reinitModules(self, filename):
        newModules = []
        newModuleIDs = {}
#        print filename
        with open(filename) as configFile:
            loadDesc = json.load(configFile)
            if type(loadDesc) is list:
                moduleDescription = loadDesc
            elif type(loadDesc) is dict:
                moduleDescription = loadDesc["modules"]
                self.fileInputList = loadDesc["inputlist"]
                self.positionalMode = True
                if not hasattr(self,"inputlist"):
                    self.inputList = self.foot["input"].keys()
                    self.inputList.sort()
                    if "modules" in self.inputList:
                        self.inputList.remove("modules")
                        
            for module in moduleDescription:
                module["parameter"] = self.parameter
                tmp = ModuleImport(module)
                if hasattr(tmp,"nrD"):
                    self.nrD = tmp.nrD
                if hasattr(tmp,"simulationSteps"):
                    self.simulationSteps = tmp.simulationSteps
                newModules.append(tmp)
                newModuleIDs[tmp.ID] = tmp
        self.modules = newModules
        self.moduleIDs = newModuleIDs

        if self.positionalMode:
            for inputName in self.fileInputList:
                self.moduleIDs[inputName] = InputModule({"ID":inputName})
        else:
            for inputName in self.inputArguments.keys():
                self.moduleIDs[inputName] = InputModule({"ID":inputName})
                
        for i in self.modules:
            i.connectInputs(self.moduleIDs)
        for i in self.modules:
            try:
                i.prepare(i.inputArguments)
            except AttributeError:
                pass
        
    def end(self):
        for i in self.modules:
            i.end()
    
    def reset(self):
        for i in self.modules:
            i.reset()

class MIMOExtension(BasicProcessingModule):
    """Groups repetitions of a single MISO module to form a MIMO module
    
    """
    def __init__(self, foot):
        """
        Setup all the parameters you need
        """
        BasicProcessingModule.__init__(self, foot)
        
        self.nrRep = foot["repititions"]
        self.splitInputs = foot["split"]
        self.moduleDescription = foot["moduleDescription"]
        self.modules = []
        for i in xrange(self.nrRep):
            self.modules.append(ModuleImport(self.moduleDescription))
            
        self.foot["input"] = self.modules[0].foot["input"]
            
    def prepare(self, antecessor):
        for module in self.modules:
            module.prepare()
            
    def __call__(self, index=0, **argIn):
        out = np.zeros(self.nrRep)
        for i in xrange(self.nrRep):
            tmpInput = argIn.copy()
            for key in self.splitInputs:
                tmpInput[key] = argIn[key][i]
            out[i] = self.modules[i](**tmpInput)
        return out
        
    def end(self):
        for i in self.modules:
            i.end()
    
    def reset(self):
        for i in self.modules:
            i.reset()

class ModularExperiment(BasicProcessingModule):
    """
    Modular Experiment
    
    Main class to set up and evaluate experiments in MPyUOSLib. This class 
    supports initialsation, evaluation and reset to prepare reevaluation of
    experiments defined by a footprint in data flow manner. The footprint 
    specifies the number of simulation steps, the random seed, the modules to 
    be used and their connections.
    """
    def __init__(self, foot, main=True):
        """
        Modular Experiment: init
        
        Store and connect processing modules described in footprint

        Input:        
        foot - Dictionary containing experiment description
        """
        self.main = main
        if self.main:
            if foot.has_key("parameter"):
                self.parameter = ParameterHandler(foot["parameter"])
            else:
                self.parameter = ParameterHandler({})
            foot["parameter"] = self.parameter
            
        BasicProcessingModule.__init__(self, foot)
        self.modules = []
        self.IDs = {}
        
        try:
            self.seed = foot["rSeed"]
        except KeyError:
            pass #TODO: Add warning about free random seed and loss of reproducibility
           
        
        
        if foot.has_key("file"):
            with open(foot["file"]) as configFile:
                moduleDescription = json.load(configFile)
        else:
            moduleDescription = foot["modules"]
        
            
        for module in moduleDescription:
            if module.has_key("parameter"):
                tmp = self.parameter
                for k, v in module["parameter"].iteritems():
                    setattr(tmp, k, v)
                module["paramter"] = tmp
            else:
                module["parameter"] = self.parameter
            tmp = ModuleImport(module)
            self.modules.append(tmp)
            self.IDs[tmp.ID] = tmp
        
        if self.main:
            for module in self.modules:
                module.connectInputs(self.IDs)
            for module in self.modules:
                module.prepare(module.inputArguments)
        if foot.has_key("nrD"):
            self.nrD = foot["nrD"]
            self.__call__ = self.callFor
        elif foot.has_key("simulationSteps"):
            self.nrD = foot["simulationSteps"]
            self.__call__ = self.callFor
        else:
            self.eventModule = self.IDs[foot["terminate"]]
            self.__call__ = self.callTerminate
            
    
    def connectInputs(self, IDs):
        """
        Group of Processing Module: connectInputs
        
        Inputs:
        names - Dictionary m
        """
        # Define group input arguments
        self.inputArguments = {}
        if self.foot.has_key("input"):
            for inputName, moduleID in self.foot["input"].iteritems():
                try:
                    self.inputArguments[inputName] = IDs[moduleID]
                except KeyError:
                    print "Error@ID:", self.ID, "-> Connection failed:", moduleID, "<->", inputName
            
        # Rename experiment input modules and add them to internal module list
        for inputName in self.inputArguments.keys():
            self.IDs[inputName] = InputModule({"ID":inputName})
            
        # Connect inputs of all experiment internal modules based on extended
        # internal module list
        for i in self.modules:
            i.connectInputs(self.IDs)

    def prepare(self, antecessor):
        """Prepare all group modules
        
        """
        for inputName,moduleOutput in self.inputArguments.iteritems():
            self.IDs[inputName].output = moduleOutput.output
        for i in self.modules:
            i.prepare(i.inputArguments)
            
        self.output = self.modules[-1].output
            
    def run(self, index):
        """Collect the input arguments to call this experiment
        
        """
        argIn = {"index":index}
        for inputName, moduleReference in self.inputArguments.iteritems():
            argIn[inputName] = moduleReference.output
        self.output = self(**argIn)
        
    
    def callTerminate(self, nrD=None, rSeed=None, index=0, **argIn):
        """
        Modular Experiment: __call__ for an event based termination of the experiment
        
        Actual evaluation of the experiment.
        """
        for inputName,moduleOutput in argIn.iteritems():
            self.IDs[inputName].output = moduleOutput
        if not self.main:
            self.prepare()
        if not nrD:
            nrD = self.nrD
        if rSeed:
            np.random.seed(rSeed)
        else:
            try:
                np.random.seed(self.seed)
            except AttributeError:
                pass
        terminate = self.eventModule.output
        simStep = 0
        while not terminate:            # Run until termination
            for module in self.modules: # Iterate all processing modules
                module.run(simStep)     # Evaluate the current simulation step
                                        # of each module
            simStep += 1                # Increment simulation step
            terminate = self.eventModule.output
            
        for module in self.modules:     # Iterate all processing modules
            module.end()                # Finish the evaluation of the experiment
        out = self.modules[-1].output
        if not self.main:
            self.reset()
        
        return out
        # Experiment done.
        
    def callFor(self, nrD=None, simulationSteps=None, rSeed=None, index=0, **argIn):
        """
        Modular Experiment: __call__ for a fixed number of steps (nrD)
        
        Actual evaluation of the experiment.
        """
        for inputName,moduleOutput in argIn.iteritems():
            self.IDs[inputName].output = moduleOutput
        if not self.main:
            self.prepare(self.inputArguments)
        
        # Handle number of simulation steps            
        if nrD is None:
            nrD = self.nrD
        if simulationSteps is None:
            nrD = self.nrD
        else:
            nrD = simulationSteps
        if type(nrD) is not int:
            try:
                nrD = self.IDs[nrD].simulationSteps
            except AttributeError:
                nrD = self.IDs[nrD].nrD
        # Handle random seed
        if rSeed:
            np.random.seed(rSeed)
        else:
            try:
                np.random.seed(self.seed)
            except AttributeError:
                pass
        for simStep in xrange(nrD):  # Iterate all simulation steps
            for module in self.modules: # Iterate all processing modules
                module.run(simStep)     # Evaluate the current simulation step
                                        # of each module
                
        for module in self.modules:     # Iterate all processing modules
            module.end()                # Finish the evaluation of the experiment
        out = self.modules[-1].output
        if not self.main:
            self.reset()
        
        return out
        # Experiment done.
        
    def end(self):
        pass
#        for module in self.modules:
#            module.end()
        
    def reset(self):
        """
        Modular Experiment: reset
        
        Prepare the experiment for reevaluation
        """
        self.output = np.zeros(1)
        for module in self.modules:                 # Iterate all processing modules
            module.reset()                          # Reset every processing module
#        for module in self.modules:
#            module.__init__(module.foot)
#        for module in self.modules:
#            module.prepare(module.inputArguments)

class ParameterHandler():
    def __init__(self,foot):
        for k,v in foot.iteritems():
            setattr(self,k,v)
        
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Modular Python UOSLib.")
    parser.add_argument("--file", help="Experiment description file")
    args = parser.parse_args()
    if args.file == None:
        from Tkinter import Tk
        from tkFileDialog import askopenfilename
        Tk().withdraw()
        filename = askopenfilename()
    else:
        filename = args.file
    with open(filename) as configFile:
        config = json.load(configFile)
    ex = ModularExperiment(config)
    ex()
