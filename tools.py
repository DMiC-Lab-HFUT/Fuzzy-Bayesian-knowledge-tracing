import numpy as np
import pandas as pd
import os
import kmeans
import gmm
from sklearn.linear_model import LogisticRegression, LinearRegression
from math import sqrt
import math
from sklearn.metrics import accuracy_score, confusion_matrix, \
    f1_score, recall_score, roc_auc_score, mean_squared_error, precision_score


def get_arrayindex(X, a):
    for i in range(len(X)):
        if X[i] == a:
            return i

def text_save(filename, data):
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')
        s = s.replace("'",'').replace(',','') +'\n'  
        file.write(s)
    file.close()
    print("保存文件成功")

def get_file(file_dir):
    file_names = []
    for root, dirs, files in os.walk(file_dir):
        file_names.append(files)
    return file_names

def Get_Average(list):
    if list:
        sum = 0
        for item in list:
            sum += item
        return sum / len(list)


def Read_X(folder):
    X = list()
    with open(folder, 'r', encoding='utf-8') as f:
        old_lines = f.readlines()
    lines = []
    for idx in range(len(old_lines)):
        if idx % 3 == 2:
            lines.append(old_lines[idx])
    for line in lines:
   
        line_list = line.strip().split('\t')

        new_line_list = list()
        for each in line_list:
            each = float(each)
            each = round(each, 3)
            if each >= 0:
                new_line_list.append(each)
        X.append(new_line_list)
    return X

def Get_X(X):

    X1 = [list(each) for each in X if list(each)]

    get_x = []
    for i in X1:
        new = []
        for j in i:
            new.append(str(j))
        get_x.append(new)
    return get_x

def kmeans_X(X, number_symbols):

    cluster_X_hat = []
    X_all = []
    for each in X:
        cluster_X_hat.append([[each_each] for each_each in each])
    for i in range(len(cluster_X_hat)):
        for each in cluster_X_hat[i]:
            X_all.append(each)

    centers, sigma1, assignments, de_dataset, dict = kmeans.k_means(X_all, number_symbols)
    return centers, sigma1, assignments, de_dataset, dict

def gmm_X(X, number_symbols):

    cluster_X_hat = []
    X_all = []
    for each in X:
        cluster_X_hat.append([[each_each] for each_each in each])
    for i in range(len(cluster_X_hat)):
        for each in cluster_X_hat[i]:
            X_all.append(each)
    X_all = np.matrix(X_all)

    Miu, Sigma, alpha = gmm.my_GMM(X_all, number_symbols)
    return Miu, Sigma, alpha


def Judge_type(qlg_y):
    if ((len(np.flatnonzero(qlg_y)) == 0) or (len(np.flatnonzero(qlg_y)) == len(qlg_y))):
        print('qlg_y_train样本类别小于2')
        return 0
    else:
        return 1

def predict(score_train, score_test, qlg_y_train, qlg_y_test, method):
    if method == 0:
        score_train, score_test = pd.DataFrame([1 if each >= 0.5 else 0 for each in score_train]), \
                                  pd.DataFrame([1 if each >= 0.5 else 0 for each in score_test])
        predict1 = np.array(score_train)
        predict2 = np.array(score_test)
    elif method == 1:
        lg = LogisticRegression() 
        lg.fit(pd.DataFrame(score_train), pd.DataFrame(qlg_y_train))
        predict1 = lg.predict(pd.DataFrame(score_train))
        predict1 = [each for each in predict1]

        lg.fit(pd.DataFrame(score_test), pd.DataFrame(qlg_y_test))
        predict2 = lg.predict(pd.DataFrame(score_test))
        predict2 = [each for each in predict2]
    elif method == 2:
        lg = LinearRegression()
        lg.fit(pd.DataFrame(score_train), pd.DataFrame(qlg_y_train))
        predict1 = lg.predict(pd.DataFrame(score_train))
        predict1 = [each for each in predict1]

        lg.fit(pd.DataFrame(score_test), pd.DataFrame(qlg_y_test))
        predict2 = lg.predict(pd.DataFrame(score_test))
        predict2 = [each for each in predict2]
    return predict1, predict2

def error_cal(target, prediction):
    error = []
    for i in range(len(prediction)):
        error.append(target[i] - prediction[i])

    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)  
        absError.append(abs(val)) 

    MSE = sum(squaredError) / len(squaredError) 
    MAE = sum(absError) / len(absError) 

    targetDeviation = []
    targetMean = sum(target) / len(target)  
    for val in target:
        targetDeviation.append((val - targetMean) * (val - targetMean))
    Variance = sum(targetDeviation) / len(targetDeviation)
    return MSE, MAE, Variance


def computeCorrelation(X, Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0, len(X)):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2
    SST = math.sqrt(varX * varY)
    return (SSR / SST) ** 2

def Rsquare(Y, Y_hat):
    yBar = np.mean(Y)
    First = 0
    Second = 0
    for i in range(len(Y_hat)):
        diffYhatY = Y_hat[i] - Y[i]
        First += diffYhatY ** 2
        diffYBarY = yBar - Y[i]
        Second += diffYBarY ** 2
    if Second == 0:
        Second += 0.0000001
    Rsquare = 1 - First / Second
    return Rsquare

def calculate(qlg_train_actual, qlg_train_pred, qlg_test_actual, qlg_test_pred,
              mse_train, mse_test, rmse_train, rmse_test, mae_train, mae_test,
              variance_train, variance_test, sigma_train, sigma_test, rsquare_train, rsquare_test):
    r1, r2, r3, r4, r5 = error_cal(qlg_train_actual, qlg_train_pred)
    r6, r7, r8, r9, r10= error_cal(qlg_test_actual, qlg_test_pred)
    r11 = Rsquare(qlg_train_actual, qlg_train_pred)
    r12 = Rsquare(qlg_test_actual, qlg_test_pred)
    mse_train.append(r1)
    mse_test.append(r6)
    rmse_train.append(r2)
    rmse_test.append(r7)
    mae_train.append(r3)
    mae_test.append(r8)
    variance_train.append(r4)
    variance_test.append(r9)
    sigma_train.append(r5)
    sigma_test.append(r10)
    rsquare_train.append(r11)
    rsquare_test.append(r12)
    return mse_train, mse_test, rmse_train, rmse_test, mae_train, mae_test, \
           variance_train, variance_test, sigma_train, sigma_test, rsquare_train, rsquare_test