#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:28:39 2020
@author: Kai and Fu

Numerical example for artical
"A Q-learning Algorithm for Discrete-time Linear-quadratic Control with Random Parameters of Unknown Distribution: Convergence and Stabilization" 
by Kai Du, Qingxin Meng and Fu Zhang

Example in Section 5.3
for critical $\rho_max$

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
#Q_star = np.array([[5,2,0], [2,2,-1], [0,-1,1]])
Q_star = np.zeros([d,d])




# coefficient of system    
def generate_coeff(discount=1):
    A = A0 + np.random.randn() * A1 + np.random.randn() * A2
    A = discount * A
    return A


error_level = 1e-4

# initialization
Q1 = 200*np.eye(d)
Q2 = 200*np.eye(d)

x = np.array([1])
error1 = np.array([1])
error2 = np.array([1])

# total time
time = 10000

disc_1 = 2.25
disc_2 = 2.4

while (x[-1]<time):
    A = generate_coeff()
    A_1 = disc_1 * A
    A_2 = disc_2 * A
    
    Phi1 = LQ.Phi(A_1,N, Q1)
    Phi2 = LQ.Phi(A_2,N, Q2)
    
    Q1 += (100/(100+x[-1])) * ( Phi1 - Q1)
    Q2 += (100/(100+x[-1])) * ( Phi2 - Q2)
    
    err1 = LQ.Error(Q1, Q_star)
    err2 = LQ.Error(Q2, Q_star)
    
    x = np.append(x, x[-1]+1)
    
    error1 = np.append(error1, err1)
    error2 = np.append(error2, err2)
#    maxim = np.append(maxim, np.trace(Q))
    
    
#    if (x[-1]>1e5):
#        break

# the period that shows in the graph
length = int(time)-1


plt.figure(figsize=(7, 5))

plt.xlim(len(x)-length, len(x))
#plt.ylim(0, 0.2)
plt.ylabel(r"$\Vert Q_t \Vert_1$")
plt.xlabel("time")    
plt.plot(x[-length:],error1[-length:], color='blue', linestyle="--",
         label=r"$\rho = 2.25$")
plt.plot(x[-length:],error2[-length:], color='red', 
         label=r"$\rho = 2.4$")
plt.legend()
#plt.show()

plt.savefig("eg3.pdf")

print(Q1)
print(error1[-1], error2[-1])
print(Q1-Q_star)
print(Q2-Q_star)