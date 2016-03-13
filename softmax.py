# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 19:54:46 2016

@author: karl.surmacz
"""
import numpy as np
import matplotlib.pyplot as plt

def softmax(vector):
    output = np.exp(vector)/np.sum(np.exp(vector))
    return output

def cross_entropy(S,L):
    D = -np.dot(S, np.log(L))
    return D

if __name__ == "__main__":
    scores = [3, 2, 0.1]
    print(softmax(scores))
    print("Check - probabilities add up to "+str(sum(softmax(scores))))
    x = np.arange(-2, 6, 0.1)
    scores = np.vstack([x, np.ones_like(x), 0.2*np.ones_like(x)])
    plt.plot(x, softmax(scores).T, linewidth=2)
    plt.show
    