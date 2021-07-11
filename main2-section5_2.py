#!/usr/bin python3
# -*- coding: utf-8 -*- 
"""
Created on May 1 09:28:39 2021
@author: Kai and Fu

Numerical example for artical
"A Q-learning Algorithm for Discrete-time Linear-quadratic Control with Random Parameters of Unknown Distribution: Convergence and Stabilization" 
by Kai Du, Qingxin Meng and Fu Zhang

Example in Section 5.2
"""

# from test.main1 import Q_star
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt

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



error_level = 1e-4


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
time = 2000


# 
error1 = np.zeros((N_trial, time))
error2 = np.zeros((N_trial, time))
error3 = np.zeros((N_trial, time))
error4 = np.zeros((N_trial,time))


A_sample = np.zeros((time, n, d))

for j in range(N_trial):

    # initialization
    Q1 = np.eye(d)
    Q2 = np.eye(d)
    Q3 = np.eye(d)
    Q4 = np.eye(d)
    '''
    Learning algorithm 
    '''
    for i in range(time):
        # sampling the coefficients
        A_sample[i,:,: ] = generate_coeff()
        
        Phi1 = LQ.Phi(A_sample[i, ],N,Q1)
        Phi2 = LQ.Phi(A_sample[i, ],N,Q2)
        Phi3 = LQ.Phi(A_sample[i, ],N,Q3)
        
        # learning iterations with 3 types of learning rate
        Q3 += (10/(10+i)) * ( Phi3 - Q3)
        Q2 += (2/(2+i)) * ( Phi2 - Q2)
        Q1 += (1/(1+i)) * ( Phi1 - Q1)
            
        error1[j, i] = LQ.Error(Q1, Q_star)
        error2[j, i] = LQ.Error(Q2, Q_star)
        error3[j, i] = LQ.Error(Q3, Q_star)
     
        # Algorithm for AQE
        N_sample = np.expand_dims(N, 0).repeat(i+1, axis = 0)
        Q4 = LQ.Sample_AQE(N_sample, A_sample[:i+1,], np.eye(d), error_level)    
        error4[j, i] = LQ.Error(Q4, Q_star)


'''
Outputs
'''
#Q_star_norm = np.linalg.norm(Q_star)

np.save('error_ALG1_1.npy', error1)
np.save('error_ALG1_.npy', error2)
np.save('error_ALG1_3.npy', error3)
np.save('error_ALG2.npy', error4)

error1_mean = error1.mean(0)
error2_mean = error2.mean(0)
error3_mean = error3.mean(0)
error4_mean = error4.mean(0)

error1_var = error1.var(0)
error2_var = error2.var(0)
error3_var = error3.var(0)
error4_var = error4.var(0)

# the period that shows in the graph
length = int(time)- 100

plt.figure(figsize=(18, 5))

plt.subplot(1,3,1)
plt.xlim(time-length, time)
plt.ylabel(r"$\Vert \,Q_t - Q^* \Vert_1$")
plt.xlabel("time")
plt.title("sigle sample trajector")
#plt.semilogy(np.arange(time - length, time), error1[0,time-length:time], color='black', linestyle="dotted", label=r"Alg1: $\alpha_t = 1/(t+1)$")
plt.semilogy(np.arange(time - length, time), error2[0, time-length:time], color='black', linestyle="-", label=r"ALGO 1")
#plt.semilogy(np.arange(time - length, time), error3[0, time-length:time], color='red', linestyle="-", label=r"Alg1: $\alpha_t = 10/(t+10)$")
plt.semilogy(np.arange(time - length, time), error4[0, time-length:time], color='red', linestyle="-.", label=r"ALGO 2")
plt.legend()

plt.subplot(1,3,2)
plt.xlim(time-length, time)
#plt.ylabel(r"mean of $\Vert \,Q_t - Q^* \Vert_1$ for 100 sample trajectors")
plt.xlabel("time")
plt.title("mean for 100 sample trajectors")
#plt.semilogy(np.arange(time - length, time), error1_mean[time-length:time], color='black', linestyle="dotted", label=r"Alg1: $\alpha_t = 1/(t+1)$")
plt.semilogy(np.arange(time - length, time), error2_mean[time-length:time], color='black', linestyle="-", label=r"ALGO 1")
#plt.semilogy(np.arange(time - length, time), error3_mean[time-length:time], color='red', linestyle="-", label=r"Alg1: $\alpha_t = 10/(t+10)$")
plt.semilogy(np.arange(time - length, time), error4_mean[time-length:time], color='red', linestyle="-.", label=r"ALGO 2")
plt.legend()

plt.subplot(1,3,3)
plt.xlim(time-length, time)
plt.xlabel("time")
plt.title(r"variance for 100 sample trajectors")
plt.plot(np.arange(time - length, time), error2_var[time-length:time], color='black', linestyle="-", label=r"ALGO 1")
plt.plot(np.arange(time - length, time), error4_var[time-length:time], color='red', linestyle="-.", label=r"ALGO 2")

plt.legend()

plt.savefig("eg2.pdf")