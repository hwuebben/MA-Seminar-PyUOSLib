# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 11:22:50 2016

@author: joschnei
"""

class SimpleController:
    def __init__(self, foot):
        self.foot = foot
        self.name = foot["name"]
        self.cnt = 0
        try:
            self.maxForce = foot["maxForce"]
        except(KeyError):
            self.maxForce = 120             # 120 N is the standard F_max in "Benchmark problems for nonlinear system identification and control using Soft Computing methods: Need and overview
        self.output = 0
        self.posErrorInt = 0
        self.posErrorOld = 0
        self.angleErrorInt = 0
        # Parameters tuned after Ziegler-Nichols
        Ku = 1400.0
        Tu = 80.0
        dt = 0.01
        self.P = 0.6*Ku
        self.I = Tu/2.0*dt
        self.D = Tu/8.0/dt

    def connectInputs(self,names):
        # connect input for external force on the cart
        self.input = names[self.foot["input"]]
        
    def run(self,index):
        # read input
        argIn = self.input.output
        # perform simulation
        result = self(argIn,index)
        # provide output
        self.output = result

    def __call__(self, argIn, index):
        posError = 50-argIn[2]
        angleError = -argIn[0]
        self.posErrorInt = 0.99*self.posErrorInt+0.01*posError
        self.angleErrorInt = 0.99*self.angleErrorInt+0.01*angleError
        result = posError*self.P+self.posErrorInt*self.I+(posError-self.posErrorOld)*self.D/0.01
        if(result > 0.1 and result < 6):
            result += 6
        elif(result < -0.1 and result > -6):
            result -= 6
#        print 'P', posError*self.P
#        print 'I', self.posErrorInt*self.I
#        print 'D', (posError-self.posErrorOld)*self.D/0.01
        self.posErrorOld = posError
#        if(index == 10):
#            return 120
#        else:
#            return 0
        return result
    
    def end(self):
        pass