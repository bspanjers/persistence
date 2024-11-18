#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:50:27 2024

@author: admin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def nw(x, X, Y, h, K=norm.pdf):
    # Arguments
    # x: evaluation points
    # X: vector (size n) with the predictor
    # Y: vector (size n) with the response variable
    # h: bandwidth
    # K: kernel
    
    # Matrix of size n x length(x)
    Kx = np.vstack([K((x - Xi) / h) / h for Xi in X])

    # Weights
    W = Kx / np.sum(Kx, axis=0)  # Column-wise sum

    # Means at x
    return np.dot(W.T, Y)  # Transpose W before the dot product


# Generate some data to test the implementation
n = 100
eps = np.random.normal(scale=2, size=n)
m = lambda x: x**2 * np.cos(x)
# m = lambda x: x - x**2 # Works equally well for other regression function
X = np.random.normal(scale=2, size=n)
Y = m(X) + eps
x_grid = np.linspace(-5, 5, num=500)

# Bandwidth
h = 0.5

# Plot data
plt.scatter(X, Y)
plt.plot(x_grid, m(x_grid), color='black', label='True regression')
plt.plot(x_grid, nw(x_grid, X, Y, h), color='red', label='Nadaraya-Watson')
plt.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def nw(x, X, Y, h, K=norm.pdf):
    # Kernel density estimation using Nadaraya-Watson method
    Kx = np.vstack([K((x - xi) / h) / h for xi in X])
    W = Kx / np.sum(Kx, axis=0)
    return np.dot(W.T, Y)

def cv_nw(X, Y, h, K=norm.pdf):
    # Cross-validation function for Nadaraya-Watson bandwidth selection
    
    # Compute pairwise differences of X
    X_diff = np.subtract.outer(X, X)
    
    # Kernel density estimation using Nadaraya-Watson method
    Kx = K(X_diff / h) / h
    
    # Weights
    W = Kx / np.sum(Kx, axis=1, keepdims=True)
    
    # Predicted values at each point
    Y_pred = np.dot(W, Y)
    
    # Compute leave-one-out prediction error
    cv_losses = ((Y - Y_pred) / (1 - K(0) / np.sum(K(X_diff / h), axis=1)))**2
    
    return np.sum(cv_losses)




def bw_cv_grid(X, Y, h_grid=None, K=norm.pdf, plot_cv=False):
    # Grid search for bandwidth selection using cross-validation
    if h_grid is None:
        h_grid =  np.ptp(X) * (np.linspace(0.05, 0.5, num=200)**2)
    obj = [cv_nw(X, Y, h, K) for h in h_grid]
    h_opt = h_grid[np.nanargmin(obj)]
    if plot_cv:
        plt.plot(h_grid, obj, marker='o')
        plt.plot(h_opt, min(obj), 'ro')  # Plot marker at minimum
        plt.xlabel('Bandwidth (h)')
        plt.ylabel('CV Loss')
        plt.title('Cross-Validation Loss for Bandwidth Selection')
        plt.show()
    return h_opt, np.argmin(obj), obj



# Set seed and generate data
n = 200
eps = np.random.normal(scale=2, size=n)
m = lambda x: x**2 + np.sin(x)
X = np.random.normal(scale=1.5, size=n)
Y = m(X) + eps
x_grid = np.linspace(-10, 10, num=500)

# Find optimal bandwidth using cross-validation
h_opt, argmin, obj = bw_cv_grid(X, Y, plot_cv=True)
print("Optimal bandwidth selected using cross-validation:", h_opt)
