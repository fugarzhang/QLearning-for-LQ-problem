#!/usr/bin python3
# -*- coding: utf-8 -*- 
"""
Created on Wed Sep 30 09:28:39 2020

@author: Kai and Fu
"""

# from test.main1 import Q_star
import numpy as np

# mapping Pi
def Pi(Q, n):
    P = Q[:n,:n] - Q[:n,n:] @ np.linalg.inv(Q[n:, n:]) @ Q[n:,:n]
    return P

# mapping Gamma
def Gamma(Q, n):
    P = - np.linalg.inv(Q[n:, n:]) @ Q[n:,:n]
    return P

# mapping Phi
def Phi(A,N,Q):
    (n, d) = A.shape 
    P = N + A.T @ Pi(Q, n) @ A
    return P

# # update Phi after an episode
# def Phi_epis(Q, epis_len=1):
#     P = np.zeros([d,d])
#     for i in range(epis_len):
#         A = generate_coeff()
#         P += Phi(A,N,Q)
#     return P/epis_len

def Error(P, Q):
    return np.linalg.norm(P-Q, ord=1)


'''
algorithm of algebra Riccati equation
'''
def Sample_AQE(N_sample, A_sample, Q_0, error_level):
    if A_sample.ndim == 2: #1 sample
        (n, d) = A_sample.shape
        A_2nd_moment = np.kron(A_sample[:,:].T, A_sample[:,:].T )
        N_1st_moment = N_sample
    elif A_sample.ndim == 3: #multi sample
        (sample_size, n, d) = A_sample.shape
        A_sample_kroneck = np.zeros((sample_size, d*d, n*n))
        for i in range(sample_size):
            A_sample_kroneck[i,:,:] = np.kron(A_sample[i,:,:].T, A_sample[i,:,:].T )
        A_2nd_moment = A_sample_kroneck.mean(0)
        N_1st_moment = N_sample.mean(0)
    else:
        print("dimension error")
    
    # Step 1 - the moments of coefficients
    A_sample_kroneck = np.zeros((sample_size, d*d, n*n))
    for i in range(sample_size):
        A_sample_kroneck[i,:,:] = np.kron(A_sample[i,:,:].T, A_sample[i,:,:].T )
    A_2nd_moment = A_sample_kroneck.mean(0)
    N_1st_moment = N_sample.mean(0)

    # Step 2 - iteration of algebra Riccati equation
    Q = np.copy(Q_0)
    Q_temp = np.copy(Q_0)
    Q =  N_1st_moment + (A_2nd_moment @ Pi(Q, n).reshape(n*n,1)).reshape(d,d)
    i = 1
    while Error(Q , Q_temp) > error_level :
        Q_temp[:,:] = Q[:,:]
        Q =  N_1st_moment + (A_2nd_moment @ Pi(Q, n).reshape(n*n,1)).reshape(d, d)
        i=i+1
        if np.linalg.norm(Q-Q_temp)>1e+30:
            Q = np.nan * np.eye(d)
            break
    return Q,i
