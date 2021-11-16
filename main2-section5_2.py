#!/usr/bin python3
# -*- coding: utf-8 -*- 
"""
Created on May 1 09:28:39 2021
@author: Kai and Fu

Numerical example for artical
"A Q-learning Algorithm for Discrete-time_data Linear-quadratic Control with Random Parameters of Unknown Distribution: Convergence and Stabilization" 
by Kai Du, Qingxin Meng and Fu Zhang

Example in Section 5.2
"""

# from test.main1 import Q_star
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import time

import LQ_function as LQ

# dimensions: n - state, m - control
n = 2
m = 1
d = n+m

rho= 1
# parameters
A0 = rho * np.array([[-1 , -0.1, -0.2],
       [ 2.6,  0.5,  0.5]])
A1 = rho * np.array([[ 0.6  ,  0.075,  0.125],
       [-0.8  ,  0.1  , -0.375]])
A2 = rho * np.array([[-0.06, -0.06,  0.02],
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



error_level = 1e-3


'''
Real value of Q_star
Iteration algorithm for Algebra Q Equation 
First and second order moments of the coefficient matrix are known. 
'''
# Q_star=np.eye(d)
# Q_star_temp = np.eye(d)
# Q_star = N + A0.T @ LQ.Pi(Q_star, n) @ A0 + A1.T @ LQ.Pi(Q_star, n) @ A1 + A2.T @ LQ.Pi(Q_star, n) @ A2
# while LQ.Error(Q_star , Q_star_temp) > error_level :
#     Q_star_temp[...] = Q_star[...]
#     Q_star = N + A0.T @ LQ.Pi(Q_star, n) @ A0 + A1.T @ LQ.Pi(Q_star, n) @ A1 + A2.T @ LQ.Pi(Q_star, n) @ A2



# trial number
N_trial = 100
# time or sample size
time_data = 2000


# 
error2 = np.zeros((N_trial, time_data))
error4 = np.zeros((N_trial,time_data))

count_AQE_itr = np.zeros((N_trial, time_data))

tick_learning = np.zeros(N_trial)
tick_AQE = np.zeros(N_trial)
tick_start=100

A_sample = np.zeros((time_data, n, d))



for j in range(N_trial):       

       print(j)
       # initialization
       Q2 = np.eye(d)

       Q4 = np.eye(d)
       '''
       Learning algorithm 
       '''

       for i in range(time_data):
              # sampling the coefficients
              A_sample[i,:,: ] = generate_coeff()

              # algorithm 1
       

       t = time.time()
       for i in range(time_data):
              Phi2 = LQ.Phi(A_sample[i, ],N,Q2)
              # learning iterations with 3 types of learning rate
              Q2 += (2/(2+i)) * ( Phi2 - Q2)
              
              error2[j, i] = LQ.Error(Q2, Q_star)

       #algorithm 2
       
       tick_learning[j]=time.time()-t

       t = time.time()
       for i in range(time_data):
              # Algorithm for AQE
              N_sample = np.expand_dims(N, 0).repeat(i+1, axis = 0)
              if np.sum(Q4)==np.nan:
                     Q4, count_AQE_itr[j,i] = LQ.Sample_AQE(N_sample, A_sample[:i+1,], np.eye(d),    error_level)
              else:
                     Q4, count_AQE_itr[j,i] = LQ.Sample_AQE(N_sample, A_sample[:i+1,], Q4, error_level)    
              error4[j, i] = LQ.Error(Q4, Q_star)

       tick_AQE[j]=time.time()-t


'''
Outputs
'''
#Q_star_norm = np.linalg.norm(Q_star)

np.save('error_ALG1_.npy', error2)
np.save('error_ALG2.npy', error4)

error2_mean = error2.mean(0)
error4_mean = error4.mean(0)

error2_var = error2.var(0)
error4_var = error4.var(0)

count_AQE_itr_mean = count_AQE_itr.mean(0)

np.save('count_AQE_itr_mean',count_AQE_itr_mean)

print(count_AQE_itr)
print(tick_learning,tick_AQE)

# the period that shows in the graph
length = int(time_data)- 100

plt.figure(figsize=(18, 5))

plt.subplot(1,3,1)
plt.xlim(time_data-length, time_data)
plt.ylabel(r"$\Vert \,Q_t - Q^* \Vert_1$")
plt.xlabel("$t$")
plt.title("single sample trajectory")
#plt.semilogy(np.arange(time_data - length, time_data), error1[0,time_data-length:time_data], color='black', linestyle="dotted", label=r"Alg1: $\alpha_t = 1/(t+1)$")
plt.semilogy(np.arange(time_data - length, time_data), error2[0, time_data-length:time_data], color='black', linestyle="-", label=r"ALGO 1")
#plt.semilogy(np.arange(time_data - length, time_data), error3[0, time_data-length:time_data], color='red', linestyle="-", label=r"Alg1: $\alpha_t = 10/(t+10)$")
plt.semilogy(np.arange(time_data - length, time_data), error4[0, time_data-length:time_data], color='red', linestyle="-.", label=r"ALGO 2")
plt.legend()

plt.subplot(1,3,2)
plt.xlim(time_data-length, time_data)
#plt.ylabel(r"mean of $\Vert \,Q_t - Q^* \Vert_1$ for 100 sample trajectors")
plt.xlabel("$t$")
plt.title(r"mean for 100 sample trajectories")
#plt.semilogy(np.arange(time_data - length, time_data), error1_mean[time_data-length:time_data], color='black', linestyle="dotted", label=r"Alg1: $\alpha_t = 1/(t+1)$")
plt.semilogy(np.arange(time_data - length, time_data), error2_mean[time_data-length:time_data], color='black', linestyle="-", label=r"ALGO 1")
#plt.semilogy(np.arange(time_data - length, time_data), error3_mean[time_data-length:time_data], color='red', linestyle="-", label=r"Alg1: $\alpha_t = 10/(t+10)$")
plt.semilogy(np.arange(time_data - length, time_data), error4_mean[time_data-length:time_data], color='red', linestyle="-.", label=r"ALGO 2")
plt.legend()

plt.subplot(1,3,3)
plt.xlim(time_data-length, time_data)
plt.xlabel("$t$")
plt.title(r"variance for 100 sample trajectories")
plt.plot(np.arange(time_data - length, time_data), error2_var[time_data-length:time_data], color='black', linestyle="-", label=r"ALGO 1")
plt.plot(np.arange(time_data - length, time_data), error4_var[time_data-length:time_data], color='red', linestyle="-.", label=r"ALGO 2")

plt.legend()

plt.savefig("eg2.pdf")