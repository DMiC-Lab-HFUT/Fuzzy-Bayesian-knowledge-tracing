'''
FuzzyBKT引入Fuzzy理论，处理0至1之间的数据集
'''
__author__ = 'lf'
from FuzzyHMM import fuzzy_hmm
import tools
import numpy as np
from sklearn.model_selection import KFold
import time
import os
from makdir import mkdir

def fuzzybkt(arg):
    # basedir = 'E:\TFS\增加实验/base\data/'
    basedir = 'E:/2#Educational data mining/1#KT/1#data\lf_data_v3/' + arg[0] + '/BKT_data/'
    # kf = KFold(n_splits=5, shuffle=True)  # shuffle=true 表示每次划分时进行重新洗牌
    # low_no_stu = 5  # 记录阈值，小于该阈值的KC被舍去

    # 认知状态的取值
    number_status = arg[2][0]  # 认知状态的个数N

    # 聚类个数
    number_clusters = arg[2][1]  # M

    # 迭代器中的参数(predict_method, forget, name_list)设置
    # predict_method = arg[0] # 0：以0.5为界；1：逻辑回归
    forget = 1  # 0：不允许遗忘(经典BKT)；1：允许遗忘
    data_name = arg[0]
    kc_name = str(arg[1])
    folder = basedir + '/' + arg[1]
    # folder = os.path.abspath(os.path.dirname(os.getcwd())) + '/' + str(data_name)
    # folder = basedir + arg[0] + '/FKT_data/' + str(arg[1]) + '/' + arg[2]
    print(data_name + '进程：文件' + kc_name + 'ing')

    #folder_X = folder + '/score/' + kc_name
    #folder_qlg = folder + '/qlg/' + kc_name
    X_train = np.array(tools.Read_X(folder=folder + '/training.txt'))  # 读取KC(log[i])的学生记录数据X
    X_test = np.array(tools.Read_X(folder=folder + '/testing.txt'))

    para = [0.3,1.7]
    #X = np.array(tools.Read_X(folder=folder_X))  # 读取KC(log[i])的学生记录数据X
    #qlg_y = np.loadtxt(folder_qlg)  # 对于每个KC，取出用作验证预测值的qlg

    #fold_MSE_train, fold_RMSE_train, fold_MAE_train, fold_Variance_train, fold_Sigma_train, fold_Rsquare_train = [],[],[],[],[],[]
    #fold_MSE_test, fold_RMSE_test, fold_MAE_test, fold_Variance_test, fold_Sigma_test, fold_Rsquare_test = [], [], [], [], [],[]

    # 得到训练数据train和test，进行交叉验证
    #for train_index, test_index in kf.split(qlg_y):
    #number_splits = 0
    #qlg_y_train, qlg_y_test = qlg_y[train_index], qlg_y[test_index]
    # 舍去样本qlg_y_train中类别小于2的情况
    ###得到训练集train和测试集test
    #X_train, X_test = X[train_index], X[test_index]
    # centers和sigma1对应于文章中的miu1和sigma1
    # de_dataset为dataset的去重版本， membership对应每一个de_dataset对每个聚类的隶属度，dict存储每个de_dataset对应的观测模糊集的隶属度
    centers, sigma1, assignments, de_dataset, dict = tools.kmeans_X(X_train, number_clusters)
    # number_symbols_new = len(centers)  # 可能由于原数据类别少等原因，导致聚类的实际个数少于number_symbols，此处对聚类个数进行更新
    train, test = tools.Get_X(X_train), tools.Get_X(X_test)  # 得到train和test

    h = fuzzy_hmm(n_state=number_status, forget_param=forget, centers=centers, sigma1=sigma1, dict=dict, para=para)
    # 使用HMM中的baum_welch()开始训练
    # train_rmses, train_maes, test_rmses, test_maes = [], [], [], []
    if train and test:
        # h.baum_welch(train, no[data_name - 1][i])
        h.baum_welch(train, test, arg)
        # train_rmses, train_maes, test_rmses, test_maes = h.baum_welch(train, test, train_rmses, train_maes, test_rmses, test_maes)
    #
    #     result_dir='E:\TFS\增加实验/base/0#result/T2FBKT_' + str(arg[3]) + 'S4C_noforget_0.1/'
    #     mkdir(result_dir )
    #     mkdir(result_dir + arg[0] + '/')
    #     mkdir(result_dir+ arg[0] + '/' + str(arg[1]) + '/')
    #     mkdir(result_dir  + arg[0] + '/' + str(arg[1]) + '/' + arg[2] + '/')
    #     with open(result_dir + arg[0] + '/' + str(arg[1]) + '/' + arg[2] + '/train_rmse.txt', 'w') as f:
    #         for x in train_rmses:
    #             f.writelines(str(x) + '\n')
    #     with open(result_dir + arg[0] + '/' + str(arg[1]) + '/' + arg[2] + '/train_mae.txt', 'w') as f:
    #         for x in train_maes:
    #             f.writelines(str(x) + '\n')
    #     with open(result_dir  + arg[0] + '/' + str(arg[1]) + '/' + arg[2] + '/test_rmse.txt', 'w') as f:
    #         for x in test_rmses:
    #             f.writelines(str(x) + '\n')
    #     with open(result_dir+ arg[0] + '/' + str(arg[1]) + '/' + arg[2] + '/test_mae.txt', 'w') as f:
    #         for x in test_maes:
    #             f.writelines(str(x) + '\n')
    #     print('文件' + str(arg[1]) + '/' + arg[2] + 'success')
    #
    #
    # # print(arg[0] + '进程：文件' + arg[1] + 'success')
