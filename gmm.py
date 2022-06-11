
from numpy import *
import numpy as np
import random
import copy

SIGMA = 6
EPS = 0.0001

def my_GMM(X, k):
    N = len(X)
    Miu = np.random.rand(k ,1)
    Posterior = mat(zeros((N ,k)))
    sigma = np.random.rand(k ,1)
    alpha = np.array([1/k] * k)
    dominator = 0
    numerator = 0
    print(sigma)
    for it in range(100):
        for i in range(N):
            dominator = 0
            for j in range(k):
                dominator = dominator + np.exp(-1.0 /(2.0 *sigma[j]) * (X[i] - Miu[j] )**2)
           
            for j in range(k):
                numerator = np.exp(-1.0 /(2.0 *sigma[j]) * (X[i] - Miu[j] )**2)
                Posterior[i ,j] = numerator / dominator
        oldMiu = copy.deepcopy(Miu)
        oldalpha = copy.deepcopy(alpha)
        oldsigma = copy.deepcopy(sigma)
   
        for j in range(k):
            numerator = 0
            dominator = 0
            for i in range(N):
                numerator = numerator + Posterior[i ,j] * X[i]
                dominator = dominator + Posterior[i ,j]
            Miu[j] = numerator /dominator
            alpha[j] = dominator /N
            tmp = 0
            for i in range(N):
                tmp = tmp + Posterior[i ,j] * (X[i] - Miu[j] )**2
          
            sigma[j] = tmp /dominator
        
        if ((abs(Miu - oldMiu)).sum() < EPS) and \
                ((abs(alpha - oldalpha)).sum() < EPS) and \
                ((abs(sigma - oldsigma)).sum() < EPS):
            print("Miu:")
            print(Miu)
            print("sigma:")
            print(sigma)
            print("alpha")
            print(alpha)
            return Miu, Sigma, alpha
            break
        print("it")
        print(it)
