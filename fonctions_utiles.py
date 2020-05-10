import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

alpha = 1
beta = 10
lambda_c = 1
lambdaPDC = 4000
puissance_min = 0
puissance_max = 23
pression_max = 68
pression_min = 45
P_M = pression_max**2
P_m = pression_min**2
D_m = 0.3
D_M = 1.5

L = np.array([[0, 0.5*10**3, None, None, 0.25*10**3], [0.5*10**3, 0, None, None, None],
            [None, None, 0, 0.5*10**3, None], [None, None, 0.5*10**3, 0, 10**3], 
            [0.25*10**3, None, None, 10**3, 0]])

Cana = [(1, 2), (3, 4), (1, 5), (5, 4)]
SC = [(2, 3)]

N = 5

def f(D, W, P):
    s1 = 0
    s2 = 0
    for x in range(len(Cana)):
        s1 += L[Cana[x][0], Cana[x][1]]*D[Cana[x][0], Cana[x][1]]
    for x in range(len(SC)):
        s2 += W[SC[x][0], SC[x][1]]
    return alpha*s1 + beta*s2

def c1_ij(D, W, P, i, j):
    return P[i] - P[j] - (lambdaPDC*L[i, j]*Q[i, j]**2)/(D0[i, j]**(5/2)+D[i, j]**(5/2))**2

def c2(D, W, P):
    return lambdaC*Q[1, 2]*np.log(P[2]/P[1]) - W - W0

def c3_ij(D, W, P, i, j):
    return P[i] - P[j]

def c4(D, W, P, i):
    return P_m - P[i]

def c5(D, W, P, i):
    return P[i] - P_M

def c6(D, W, P, i, j):
    return D_m - D[i, j]

def c7(D, W, P, i, j):
    return D[i, j] - D_M

def c8(D, W, P):
    return puissance_min - W

def c9(D, W, P):
    return W - puissance_max


def ce(D, W, P):
    return np.array([P[i-1] - P[j-1] - lambdaPDC*L[i, j]*Q[i-1, j-1]**2 for i,j in Cana] + [lambdaC*Q[1, 2]*np.log(P[2]/P[1]) - W])

def ci(D, W, P):
    return np.concatenate(np.array([P[1] - P[2]]), P_m - P, P - P_M, D_m - D, D - D_M, puissance_min - W, puissance_max - W)