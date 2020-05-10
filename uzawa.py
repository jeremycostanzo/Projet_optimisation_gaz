import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def uzawa_fixed_step(fun, grad_fun, ce, jac_ce, ci, jac_ci, x0, l, rho, lambda0, mu0, max_iter, epsilon_grad_L = 1e-8):
	k = 0
	xk = x0
	lambdak = lambda0
        muk = mu0
	grad_Lagrangienk_xk = grad_fun(xk) + np.matmul(np.transpose(jac_ce(xk)), lambdak) + np.matmul(np.transpose(jac_ci(xk)), muk)
	while ((k<max_iter) and (np.linalg.norm(grad_Lagrangienk_xk)>epsilon_grad_L)):
		grad_Lagrangienk_xk = grad_fun(xk) + np.matmul(np.transpose(jac_ce(xk)), lambdak) + np.matmul(np.transpose(jac_ci(xk)), muk)
		pk = -grad_Lagrangienk_xk
		xk = xk + l*pk;        
		muk = np.maximum(0, muk + rho*ci(xk))
                lambdak = lambdak + rho*ce(xk)
		k = k + 1
	print("Nombre d'iterations : ", k)
	print("lambdak : ", lambdak)
        print("muk : ", muk)
	return xk
