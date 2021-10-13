#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 15:24:49 2021

@author: Yukai Yang
"""

#%% Q2 Description
'''
Numerically verify this property by simulating ||X||^2 in dimensions 
d ∈ {10,100,1000,10000} using a sample size of n = 1000.
'''

#%% import packages
import numpy as np;
import matplotlib.pyplot as plt;

#%% Do the math work
n = 1000;
d = [10,100,1000,10000];
output = [];
for dim in d:
    ## WARNING: due to the poor efficiency of this code, it may take an extremely long time on 
    ## to run the last iterations with d = 10,000. Pls be patient...
    
    
    '''
     Multivariate Random Normal Required!
    '''
    
    #np_x = np.array([np.random.normal(scale=1/dim,size = dim) for i in range(n)]);
    #np_x = np.array([np.random.normal(scale=1/dim) for i in range(n*dim)]).reshape(n,dim)
    np_x = np.array(np.random.multivariate_normal([0 for i in range(dim)], np.identity(dim)/dim,n));
    
    np_X = np.array([sum(np_x[i][j]**2 for j in range(np_x.shape[1])) for i in range(np_x.shape[0])]);
    #np_X = np_x**2;
    print("d = {}, n = {}".format(dim,n));
    print("X.mean: {:.5f}\nX.std: {:.5f}\nX^2.mean: {:.8f}\nX^2.std: {:.8f}".format(np_x.mean(),np_x.std(),np_X.mean(),np_X.std()));
    print("X^2.std/(1/sqrt(d)) = {:.4f}".format(np_X.std()*dim**(1/2)));
    print();
    output.append(np_X.std()*dim**(1/2));

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
d_str = ['10', '100','1000','10000'];
ax.bar(d_str,output)
plt.show()