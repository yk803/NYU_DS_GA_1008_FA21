#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:10:53 2021

@author: zhouhaiqing
"""

#%%
import numpy as np;
#import pandas as pd;
import matplotlib.pyplot as plt;

#%%
import itertools;
#import random as rand;
#from sklearn.linear_model import LogisticRegression as LR; #usually lr == learning rate? so I'll use LR here xD
from sklearn.linear_model import LinearRegression as LR;
from sklearn.metrics import mean_squared_error as mse;
from sklearn import preprocessing;
#%% Generate data / Prepare parameters

dimensions = np.arange(5,14);
n0 = 5;
#n0 = 10;
n = 2**n0; ##default n
K = 10;
Ks = np.zeros(dimensions.shape[0]);

def z_all(d): #generate a plus-minus one
    return ["".join(seq) for seq in itertools.product("01", repeat=d)]

def z_helper(x):
    return ''.join(['0' if d < 0 else '1' for d in x])

def g_helper(d):
    return np.random.choice([-1,1],2**d);



for d in dimensions:
    print("d={}".format(d));
    print();
    K_iters = np.zeros(K);

    x_train = np.random.uniform(-1,1,[n,d]);
    x_test = np.random.uniform(-1,1,[n,d]);
    y_train = np.zeros(n);
    y_test = np.zeros(n);
    zs = z_all(d);
    for k in range(1,K+1):
        print("iter: {}".format(k),end='\t  ');
        g = g_helper(d);
        for i in range(len(x_train)):
            x0 = x_train[i];
            encode = g[zs.index(z_helper(x0))];
            distance = abs(0.5-np.max(x_train[i]));
            y_train[i] = distance*encode;
            x1 = x_test[i];
            encode = g[zs.index(z_helper(x1))];
            distance = abs(0.5-np.max(x_test[i]));
            y_test[i] = distance*encode;
    
        model = LR().fit(x_train,y_train);
        print('Generalisation error: {}'.format(mse(y_test,model.predict(x_test))/np.std(y_test)));
        K_iters[k-1] = mse(y_test,model.predict(x_test))/np.std(y_test);
    print('Aveage genealisation error: {}'.format(np.mean(K_iters)));
    Ks[d-5] = np.mean(K_iters);
    print('\n');
plt.bar(dimensions,Ks);
plt.title("n="+str(n0));
plt.show();


'''
### I didnt figure out a coding-efficient way to avoid the super nested for loop.
### So to visualize results with different parameters, I'll have to use a nested loop here.
ns= [2**i for i in range(5,17)];
for n in ns:
    print("n={}".format(n));
    for d in dimensions:
        print("d={}".format(d));
        print();
        K_iters = np.zeros(K);
        Ks = np.zeros(dimensions.shape[0]);
        x_train = np.random.uniform(-1,1,[n,d]);
        x_test = np.random.uniform(-1,1,[n,d]);
        y_train = np.zeros(n);
        y_test = np.zeros(n);
        zs = z_all(d);
        for k in range(1,K+1):
            print("iter: {}".format(k),end='\t');
            g = g_helper(d);
        
            for i in range(len(x_train)):
                x0 = x_train[i];
                encode = g[zs.index(z_helper(x0))];
                distance = abs(0.5-np.max(x_train[i]));
                y_train[i] = distance*encode;
                x1 = x_test[i];
                encode = g[zs.index(z_helper(x1))];
                distance = abs(0.5-np.max(x_test[i]));
                y_test[i] = distance*encode;
        
            model = LR().fit(x_train,y_train);
            print('Generalisation error: {}'.format(mse(y_test,model.predict(x_test))/np.std(y_test)));
            K_iters[k-1] = mse(y_test,model.predict(x_test))/np.std(y_test);
        print('Aveage genealisation error: {}'.format(np.mean(K_iters)));
        Ks[d-5] = np.mean(K_iters);
        print('\n');
    plt.bar(dimensions,K_iters);
    plt.title("d = {}".format(d));
    plt.show();
'''
    
