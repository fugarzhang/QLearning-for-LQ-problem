#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:28:39 2020
@author: Kai and Fu

Numerical example for artical
"A Q-learning Algorithm for Discrete-time Linear-quadratic Control with Random Parameters of Unknown Distribution: Convergence and Stabilization" 
by Kai Du, Qingxin Meng and Fu Zhang

Example in Section 5.4

"""

#from main2 import A_sample, N_sample
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt

import LQ_function as LQ

# dimensions
n = 2
m = 1

d = n+m

# parameters
A1 = np.array([[-5, 2],
               [2, 3]])
A2 = np.array([[0, -1],
               [-4, 7]])
A3 = np.array([[-2, 3],
               [6, 0]])

B1 = np.array([[-1],[1]])

B2 = np.array([[-1],[0]])

N = np.array([[1, 0, 0],
       [0, 1, 0],
       [0, 0, 0]])



# coefficient of system    
def generate_coeff(discount=0.01):
    w = np.random.rand(6)
    A = (np.exp(w[0]*w[1]))*A1 - (np.sin(w[1]))*A2 - (np.sqrt(w[1]+w[2]))*A3
    B = (w[3] - w[4])*B1 + (np.cos(w[3]))*B2
    coeff = np.hstack((A,B))
    coeff = discount * coeff
    return coeff



# initialization

# total time
time = 50

#error level
error_level = 1e-4



# initial

A_sample = np.zeros((time, n, d))
N_sample = np.expand_dims(N, 0).repeat(time, axis = 0)

X_adapt = np.zeros((n, time))
X_star = np.zeros((n, time))
X_AQE_1 = np.zeros((n, time))
X_AQE_2 = np.zeros((n, time))

norm_adapt = np.array(time)
norm_star = np.array(time)
norm_AQE_1 = np.array(time)
norm_AQE_2 = np.array(time)

#initial state
X_ini = np.array([[1],[0]])

X_adapt[:, [0]] = np.copy(X_ini)
X_star[:, [0]] = np.copy(X_ini)
X_AQE_1[:, [0]] = np.copy(X_ini)
X_AQE_2[:, [0]] = np.copy(X_ini)

Q = np.eye(d)
Q_AQE_1 = np.eye(d)
Q_AQE_2 = np.eye(d)
control_adapt = np.zeros((m, n))
control_adapt_AQE_1 = np.zeros((m, n))
control_adapt_AQE_2 = np.zeros((m, n))

for i in range(time-1):
    A_sample[i, ] = generate_coeff(0.25)
    Phi1 = LQ.Phi( A_sample[i,], N_sample[i,], Q)    
    Q += (2/(2+i)) * ( Phi1 - Q)
    
    #update X_adapt
    control_adapt = LQ.Gamma(Q, n)
    X_adapt[:, i+1] = (A_sample[i, :, :n] + A_sample[i,:,n:] @ control_adapt) @ X_adapt[:,i]    
    
    #update X_star
    X_star[:,i+1] = A_sample[i, :,:n] @ X_star[:,i]
    
    #adapted control by AQE algorithm: update every 5 step
    if  i and (not i%5) :
        Q_AQE_1 = LQ.Sample_AQE(N_sample[:i+1], A_sample[0:i+1, ], Q_AQE_1, error_level)
        if not np.count_nonzero(np.isnan(Q_AQE_1)):
            control_adapt_AQE_1 = LQ.Gamma(Q_AQE_1, n)
        else:
            Q_AQE_1 = np.copy(np.eye(d))
            
    X_AQE_1[ :, i+1] = A_sample[i, :, :] @ np.vstack((np.eye(n), control_adapt_AQE_1)) @ X_AQE_1[:, i]
        
#    #adapted control by AQE algorithm: update every 20 step
    if  i and (not i%20) :
        Q_AQE_2 = LQ.Sample_AQE(N_sample[:i+1], A_sample[0:i+1, ], Q_AQE_2, error_level)
        if not np.count_nonzero(np.isnan(Q_AQE_2)):
            control_adapt_AQE_2 = LQ.Gamma(Q_AQE_2, n)
        else:
            Q_AQE_2 = np.copy(np.eye(d))
            
    X_AQE_2[ :, i+1] = A_sample[i, :, :] @ np.vstack((np.eye(n), control_adapt_AQE_2)) @ X_AQE_2[:, i]
        

# compute the norm
norm_adapt = np.linalg.norm(X_adapt, axis =0)
norm_star = np.linalg.norm(X_star, axis =0)
norm_AQE_1 = np.linalg.norm(X_AQE_1, axis =0)
norm_AQE_2 = np.linalg.norm(X_AQE_2, axis =0)


#the period that shows in the graph
length =  int(time-1)


plt.figure(figsize=(4,4))

plt.title(r"$x_0 = [1,0]^{T}$")
plt.xlim(time-length, time)
#plt.ylim(0, 0.2)
plt.ylabel(r"$\Vert x_t \Vert$")
plt.xlabel("time")    
plt.semilogy(np.arange(time-length, time), norm_star[-length:], color='red', linestyle="--",
          label=r"No control")
plt.semilogy(np.arange(time-length, time), norm_adapt[-length:], color='blue', 
          label=r"ALGO 1")

plt.semilogy(np.arange(time-length, time), norm_AQE_1[-length:], color='orange', linestyle=":",
          label=r"ALGO 2: 5 steps")
plt.semilogy(np.arange(time-length, time), norm_AQE_2[-length:], color='purple', linestyle="-.",
          label=r"ALGO 2: 20 steps")
plt.legend(loc = 'lower left')




plt.tight_layout()
plt.savefig("eg4-1.pdf")


