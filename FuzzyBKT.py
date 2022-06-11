
__author__ = 'lf'
from FuzzyHMM import fuzzy_hmm
import tools
import numpy as np
from sklearn.model_selection import KFold
import time
import os
from makdir import mkdir

def fuzzybkt(arg):

    basedir = 'data/' + arg[0] + '/BKT_data/'
  
    number_status = arg[2][0] 
    number_clusters = arg[2][1] 

    forget = 1  
    data_name = arg[0]
    kc_name = str(arg[1])
    folder = basedir + '/' + arg[1]
    X_train = np.array(tools.Read_X(folder=folder + '/training.txt')) 
    X_test = np.array(tools.Read_X(folder=folder + '/testing.txt'))

    para = [0.3,1.7]
  
    centers, sigma1, assignments, de_dataset, dict = tools.kmeans_X(X_train, number_clusters)
    train, test = tools.Get_X(X_train), tools.Get_X(X_test)

    h = fuzzy_hmm(n_state=number_status, forget_param=forget, centers=centers, sigma1=sigma1, dict=dict, para=para)
 
    if train and test:
     
        h.baum_welch(train, test, arg)
    
