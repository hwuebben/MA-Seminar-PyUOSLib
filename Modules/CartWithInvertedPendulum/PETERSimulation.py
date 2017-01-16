# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 13:55:29 2016

@author: joschnei
"""

from __future__ import division
import pylab as plt

from CartWithInvertedPendulumEasy import CartWithInvertedPendulumEasy

if __name__ == "__main__":
    sim = CartWithInvertedPendulumEasy({"name":"Balance","state":[3.14159,0,0,0.1],"dt":0.01})
    state = []
    nrD = 10000
    for i in range(nrD):
        state.append(sim(0,i))
    plt.plot(range(nrD),state)
    plt.legend(['theta','dtheta','x','dx'])
    plt.show()