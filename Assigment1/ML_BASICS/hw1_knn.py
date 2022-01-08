# -*- coding: utf-8 -*-
"""
Created on  

@author: fame
"""

import numpy as np


def compute_euclidean_distances(X, Y):
    """
    Compute the Euclidean distance between two matricess X and Y  
    Input:
    X: N-by-D numpy array 
    Y: M-by-D numpy array 
    
    Should return dist: M-by-N numpy array   
    """
    dist = []
    for y in Y:
        dist.append(np.sum((np.square(X - y)), axis=1))
    dist = np.array(dist)
    return dist


def predict_labels(dists, labels, k=1):
    """
    Given a Euclidean distance matrix and associated training labels predict a label for each test point.
    Input:
    dists: M-by-N numpy array 
    labels: is a N dimensional numpy array
    
    Should return  pred_labels: M dimensional numpy array
    """
    predictions = np.argsort(dists, axis=1)[:, :k]
    pred_labels = np.argmax(np.array([np.bincount(row.astype(int), minlength=10) for row in labels[predictions]]), axis=1)
    return pred_labels, predictions
