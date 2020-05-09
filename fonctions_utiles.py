import numpy as np

alpha = 1
beta = 10
lambda_c = 1
lambdaPDC = 4000
puissance_min = 0
puissance_max = 23

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

def c2_ij(D, W, P, i, j):
    return lambdaC*Q[i, j]*np.log(P[j]/P[i]) - W[i, j] - W0[i, j]

def c3_ij(D, W, P, i, j):
    return P[i] - P[j]

def c4(D, W, P):
    return 45 - P

def c4bis(D, W, P):
    return P - 68

def c5(D, W, P, i, j):
    return 0.3 - [i, j]

def c5bis_ij(D, W, P, i, j):
    return D[i, j] - 1.5

def c6_ij(D, W, P, i, j):
    return puissance_min - W[i, j]

def c6bis_ij(D, W, P, i, j):
    return W[i, j] - puissance_max

