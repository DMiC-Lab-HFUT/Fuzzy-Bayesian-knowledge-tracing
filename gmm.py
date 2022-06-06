
from numpy import *
import numpy as np
import random
import copy

SIGMA = 6
EPS = 0.0001

# EM算法
def my_GMM(X, k):
    N = len(X)
    Miu = np.random.rand(k ,1)
    Posterior = mat(zeros((N ,k)))
    sigma = np.random.rand(k ,1)
    # sigma[0 ] =1
    # sigma[1]=2
    # alpha = np.random.rand(k ,1)
    # alpha[0] = 0.1
    # alpha[1] = 0.9
    alpha = np.array([1/k] * k)
    dominator = 0
    numerator = 0
    # 先求后验概率
    print(sigma)
    for it in range(100):
        for i in range(N):
            dominator = 0
            for j in range(k):
                dominator = dominator + np.exp(-1.0 /(2.0 *sigma[j]) * (X[i] - Miu[j] )**2)
            # print -1.0/(2.0*sigma[j]),(X[i] - Miu[j])**2,-1.0/(2.0*sigma[j]) * (X[i] - Miu[j])**2,np.exp(-1.0/(2.0*sigma[j]) * (X[i] - Miu[j])**2)
            # return
            for j in range(k):
                numerator = np.exp(-1.0 /(2.0 *sigma[j]) * (X[i] - Miu[j] )**2)
                Posterior[i ,j] = numerator / dominator
        oldMiu = copy.deepcopy(Miu)
        oldalpha = copy.deepcopy(alpha)
        oldsigma = copy.deepcopy(sigma)
        # 最大化
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
            # print tmp,Posterior[i,j],(X[i] - Miu[j])**2
            sigma[j] = tmp /dominator
            # print (tmp, dominator, sigma[j])
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
