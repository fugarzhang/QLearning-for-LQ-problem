#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
Created on Wed Sep 30 09:28:39 2020
@author: Kai and Fu

Numerical example for artical
"A Q-learning Algorithm for Discrete-time Linear-quadratic Control with Random Parameters of Unknown Distribution: Convergence and Stabilization" 
by Kai Du, Qingxin Meng and Fu Zhang

Example in Section 5.1
"""

import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt

import LQ_function as LQ

# dimensions
n = 2
m = 1

d = n+m

# parameters
A0 = np.array([[-1 , -0.1, -0.2],
       [ 2.6,  0.5,  0.5]])
A1 = np.array([[ 0.6  ,  0.075,  0.125],
       [-0.8  ,  0.1  , -0.375]])
A2 = np.array([[-0.06, -0.06,  0.02],
       [ 0.2 ,  0.23, -0.09]])
N = np.array([[ 3.11    ,  1.5626  , -0.2798  ],
       [ 1.5626  ,  1.816175, -1.021425],
       [-0.2798  , -1.021425,  0.91585 ]])

# Q_star
Q_star = np.array([[5,2,0], [2,2,-1], [0,-1,1]])


# coefficient of system    
def generate_coeff(discount=1):
    A = A0 + np.random.randn() * A1 + np.random.randn() * A2
    A = discount * A
    return A

error_level = 1e-4

# initialization
Q1 = np.eye(d)
Q2 = np.eye(d)
Q3 = np.eye(d)

x = np.array([1])
error1 = np.array([1])
error2 = np.array([1])
error3 = np.array([1])
maxim = np.array([1])

# total time
time = 2000

while (x[-1]<time):
    A = generate_coeff()
    Phi1 = LQ.Phi(A,N,Q1)
    Phi2 = LQ.Phi(A,N,Q2)
    Phi3 = LQ.Phi(A,N,Q3)
    
    Q1 += (10/(10+x[-1])) * ( Phi1 - Q1)
    Q2 += (2/(2+x[-1])) * ( Phi2 - Q2)
    Q3 += (1/(1+x[-1])) * ( Phi3 - Q3)
    
    err1 = LQ.Error(Q1, Q_star)
    err2 = LQ.Error(Q2, Q_star)
    err3 = LQ.Error(Q3, Q_star)
    
    x = np.append(x, x[-1]+1)

    error1 = np.append(error1, err1)
    error2 = np.append(error2, err2)
    error3 = np.append(error3, err3)
#    maxim = np.append(maxim, np.trace(Q))



# the period that shows in the graph
length = int(time)-100


plt.figure(figsize=(7, 5))

plt.xlim(len(x)-length, len(x))
#plt.ylim(0, 0.2)
plt.ylabel(r"$\Vert \,Q_t - Q^* \Vert_1$")
plt.xlabel("time")
plt.semilogy(x[-length:],error3[-length:], color='black', linewidth="3",
         label=r"$\alpha_t = 1/(t+1)$")
plt.semilogy(x[-length:],error1[-length:], color='red', linestyle="--",
         label=r"$\alpha_t = 10/(t+10)$")
plt.semilogy(x[-length:],error2[-length:], color='blue', linewidth="1",
         label=r"$\alpha_t = 2/(t+2)$")

plt.legend()

plt.savefig("eg1.pdf")


print(Q1)
print(error1[-1], error2[-1], error3[-1])
print(Q1-Q_star)
print(Q2-Q_star)
print(Q3-Q_star)