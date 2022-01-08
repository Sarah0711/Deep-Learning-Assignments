# -*- coding: utf-8 -*-
"""
Created on  

@author: Jan Scheffczyk
"""

import numpy as np 
import matplotlib.pyplot as plt

 

def predict(X,W,b):  
    """
    implement the function h(x, W, b) here  
    X: N-by-D array of training data 
    W: D dimensional array of weights
    b: scalar bias

    Should return a N dimensional array  
    """
    #print("Sigmoid", W.shape, W[np.newaxis, :].shape)
    return np.squeeze(sigmoid( W[np.newaxis, :] @ X.T + b ))


 
def sigmoid(a): 
    """
    implement the sigmoid here
    a: N dimensional numpy array

    Should return a N dimensional array  
    """
    return (1 /( 1 + np.exp(-a)))
    

def l2loss(X,y,W,b):  
    """
    implement the L2 loss function
    X: N-by-D array of training data 
    y: N dimensional numpy array of labels
    W: D dimensional array of weights
    b: scalar bias

    Should return three variables: (i) the l2 loss: scalar, (ii) the gradient with respect to W, (iii) the gradient with respect to b
     """
    sigma_k = predict(X, W, b) # 

    loss = y - sigma_k
    loss = loss * loss
    loss = np.sum(loss)

    dw = y - sigma_k
    dw = dw * -sigma_k
    dw = dw * (1-sigma_k)
    db = 2 * dw
    dw = 2 * dw @ X

    return loss, dw, db


def train(X,y,W,b, num_iters=1000, eta=0.1):  
    """
    implement the gradient descent here
    X: N-by-D array of training data 
    y: N dimensional numpy array of labels
    W: D dimensional array of weights
    b: scalar bias
    num_iters: (optional) number of steps to take when optimizing
    eta: (optional)  the stepsize for the gradient descent

    Should return the final values of W and b    
     """

    losses = []
    for i in range(num_iters):
        loss, dw, db = l2loss(X, y, W, b)
        W = W - eta * dw
        b = b - eta * db
        losses.append(loss)


    plt.plot(range(num_iters), losses)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()


    return W, b






















 
