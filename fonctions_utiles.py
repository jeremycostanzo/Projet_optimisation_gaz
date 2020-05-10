import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

alpha = 1
beta = 10
lambdaC= 1
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

D0 = np.array([[0, 1.5, None, None, 1.5], [1.5, 0, None, None, None],
            [None, None, 0, 1.5, None], [None, None, 1.5, 0, 1.5], 
            [1.5, None, None, 1.5, 0]])

D_in = np.array([[0, 0.395, None, None, 0.5], [0.395, 0, None, None, None],
            [None, None, 0, 0.395, None], [None, None, 0.395, 0, 0.5], 
            [0.5, None, None, 0.5, 0]])

Q = np.array([[2, 0.5, None, None, 0.45], [-0.5, 0, 0.5, None, None],
            [None, -0.5, 0, 0.5, None], [None, None, -0.5, 1, 0.45], 
            [-0.45, None, None, -0.45, 0]])

P0 = np.array([46.32, 45.66, 45.66, 45, 46.05])**2

W = 0

Cana = [(1, 2), (3, 4), (1, 5), (5, 4)]
SC = [(2, 3)]

N = 5

D0_uzawa = [D0[i-1, j-1] for i, j in Cana]


def f(x):
    D = x[0]
    W = x[1]
    P = x[2]
    s1 = 0
    s2 = 0
    for x in range(len(Cana)):
        s1 += L[Cana[x][0], Cana[x][1]]*D[Cana[x][0], Cana[x][1]]
    for x in range(len(SC)):
        s2 += W[SC[x][0], SC[x][1]]
    return alpha*s1 + beta*s2

def grad_f(x):
    D = x[0]
    W = x[1]
    P = x[2]
    return np.transpose(np.array([alpha*[L[0, 1], L[2, 3], L[0, 4], L[4, 3]] + [beta] + [0, 0, 0, 0, 0]]))


def ce(x):
    D = x[0]
    W = x[1]
    P = x[2]
    return np.array([P[i-1] - P[j-1] - lambdaPDC*L[i, j]*Q[i-1, j-1]**2 for i,j in Cana] + [lambdaC*Q[1, 2]*np.log(P[2]/P[1]) - W])

def ci(x):
    D = x[0]
    W = x[1]
    P = x[2]
    return np.concatenate(np.array([P[1] - P[2]]), P_m - P, P - P_M, D_m - D, D - D_M, puissance_min - W, puissance_max - W)


def Jce(x):
    D = x[0]
    W = x[1]
    P = x[2]
    return np.array([[5*lambdaPDC*L[0, 1]*Q[0, 1]**2*D[0]**(3/2)/(D_in[0, 1]**(5/2) + D[0]**(5/2))**2, 0, 0, 0, 0, 1, -1, 0, 0, 0],
                    [0, 5*lambdaPDC*L[2, 3]*Q[2, 3]**2*D[1]**(3/2)/(D_in[2, 3]**(5/2) + D[1]**(5/2))**2, 0, 0, 0, 0, 0, 1, -1, 0],
                    [0, 0, 5*lambdaPDC*L[0, 4]*Q[0, 4]**2*D[2]**(3/2)/(D_in[0, 4]**(5/2) + D[2]**(5/2))**2, 0, 0, 1, 0, 0, 0, 0, -1],
                    [0, 0, 0, 5*lambdaPDC*L[4, 3]*Q[4, 3]**2*D[3]**(3/2)/(D_in[4, 3]**(5/2) + D[3]**(5/2))**2, 0, 0, 0, 0, 0, -1, 1],
                    [0, 0, 0, 0, -1, 0, - lambdaC*Q[1, 2]/P[1], lambdaC*Q[1, 2]/P[2], 0, 0, 0]])

def Jci(x):
    return np.array([[0, 0, 0, 0, 0, 0, 1, -1, 0, 0],
                    [0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, -1, 0]
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
