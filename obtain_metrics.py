from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import metrics
from makdir import mkdir
import numpy as np

def obtain_rmse(actual, pred):
    return sqrt(mean_squared_error(actual, pred))

def obtain_mae(actual, pred):
    return mean_absolute_error(actual, pred)

def obtain_auc(actual, pred):
    fpr, tpr, thresholds = metrics.roc_curve(actual, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

def obtain_r2(actual, pred):
    return r2_score(actual, pred)

def obtain_metrics(c_actual, c_pred, b_actual, b_pred):
    c_actual = np.array(c_actual).astype(float).tolist()
    b_actual = np.array(b_actual).astype(int).tolist()

    cc_rmse = obtain_rmse(c_actual, c_pred)
    cc_mae = obtain_mae(c_actual, c_pred)
    cb_rmse = obtain_rmse(c_actual, b_pred)
    cb_mae = obtain_mae(c_actual, b_pred)
    bb_rmse = obtain_rmse(b_actual, b_pred)
    bb_mae = obtain_mae(b_actual, b_pred)
    bb_auc = obtain_auc(b_actual, b_pred)
    return cc_rmse,cc_mae,cb_rmse,cb_mae,bb_rmse,bb_mae,bb_auc

def write_matrics(tr_cc_rmse, tr_cc_mae, tr_cb_rmse, tr_cb_mae, tr_bb_rmse, tr_bb_mae, tr_bb_auc,
                  te_cc_rmse, te_cc_mae, te_cb_rmse, te_cb_mae, te_bb_rmse, te_bb_mae, te_bb_auc,
                  base_dir, args):
    #write train metrics
    mkdir(base_dir)
    mkdir(base_dir + args[0] + '/')
    mkdir(base_dir + args[0] + '/' + str(args[1]) + '/')

    with open(base_dir + args[0] + '/' + str(args[1]) + '/tr_cc_rmse.txt', 'a') as f:
        for each in tr_cc_rmse:
            f.writelines(str(each) + '\n')
    with open(base_dir + args[0] + '/' + str(args[1]) + '/tr_cc_mae.txt', 'a') as f:
        for each in tr_cc_mae:
            f.writelines(str(each) + '\n')
    with open(base_dir + args[0] + '/' + str(args[1])  + '/tr_cb_rmse.txt', 'a') as f:
        for each in tr_cb_rmse:
            f.writelines(str(each) + '\n')
    with open(base_dir + args[0] + '/' + str(args[1]) + '/tr_cb_mae.txt', 'a') as f:
        for each in tr_cb_mae:
            f.writelines(str(each) + '\n')
    with open(base_dir + args[0] + '/' + str(args[1])  + '/tr_bb_rmse.txt', 'a') as f:
        for each in tr_bb_rmse:
            f.writelines(str(each) + '\n')
    with open(base_dir + args[0] + '/' + str(args[1]) + '/tr_bb_mae.txt', 'a') as f:
        for each in tr_bb_mae:
            f.writelines(str(each) + '\n')
    with open(base_dir + args[0] + '/' + str(args[1]) + '/tr_bb_auc.txt', 'a') as f:
        for each in tr_bb_auc:
            f.writelines(str(each) + '\n')
    # write test metrics
    with open(base_dir + args[0] + '/' + str(args[1])+ '/te_cc_rmse.txt', 'a') as f:
        for each in te_cc_rmse:
            f.writelines(str(each) + '\n')
    with open(base_dir + args[0] + '/' + str(args[1]) + '/te_cc_mae.txt', 'a') as f:
        for each in te_cc_mae:
            f.writelines(str(each) + '\n')
    with open(base_dir + args[0] + '/' + str(args[1]) + '/te_cb_rmse.txt', 'a') as f:
        for each in te_cb_rmse:
            f.writelines(str(each) + '\n')
    with open(base_dir + args[0] + '/' + str(args[1])  + '/te_cb_mae.txt', 'a') as f:
        for each in te_cb_mae:
            f.writelines(str(each) + '\n')
    with open(base_dir + args[0] + '/' + str(args[1]) + '/te_bb_rmse.txt', 'a') as f:
        for each in te_bb_rmse:
            f.writelines(str(each) + '\n')
    with open(base_dir + args[0] + '/' + str(args[1]) + '/te_bb_mae.txt', 'a') as f:
        for each in te_bb_mae:
            f.writelines(str(each) + '\n')
    with open(base_dir + args[0] + '/' + str(args[1]) + '/te_bb_auc.txt', 'a') as f:
        for each in te_bb_auc:
            f.writelines(str(each) + '\n')

    print("write success")

