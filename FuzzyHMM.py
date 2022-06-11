import numpy as np
from numpy import random as rand
import calculate as cal
import time
from sklearn import metrics
import tools
from obtain_metrics import *
from itertools import chain
from tqdm import tqdm


class fuzzy_hmm:
    def __init__(self, n_state, forget_param, centers, sigma1, dict,para):
       
        self.N = n_state
        self.miu2 = np.array(range(self.N)).astype(float) / (self.N - 1) 

        self.sigma2_low, self.sigma2_high = np.ones(self.N).astype(float) / (self.N), \
                                            np.ones(self.N).astype(float) / (self.N) 
       
        raw_Pi = np.array([1] * self.N)
        Pi = raw_Pi.astype(float) / raw_Pi.sum()
        self.Pi_low, self.Pi_high = Pi, Pi
       
        self.miu1 = [each[0] for each in centers] 
        self.M = len(centers)
        self.sigma1 = sigma1 

        self.dict1 = dict 

        self.w = [each / sum(np.ones(self.M)) for each in np.ones(self.M)]
        self.miu3 = np.array(range(self.N)).astype(float) / (self.N - 1)

        self.miu3 = sorted(self.miu3)
        self.sigma3 = np.ones(self.N)
       
        self.k = para
        self.sigma3_low = self.sigma3 * self.k[0]
        self.sigma3_high = self.sigma3 * self.k[1]
        self.bij_low, self.bij_high = np.zeros([self.N, self.M], float), np.zeros([self.N, self.M], float)

        raw_T = rand.uniform(0, 1, self.N * self.N).reshape(self.N, self.N)
       
        if forget_param == 0:
            raw_T = np.triu(raw_T)
        self.T_low, self.T_high = raw_T.astype(float) / raw_T.sum(axis=1, keepdims=True), \
                                  raw_T.astype(float) / raw_T.sum(axis=1, keepdims=True)
      
    def Obtain_Eij(self):
        bij_low, bij_high = np.zeros([self.N, self.M], float), np.zeros([self.N, self.M], float)
        for i in range(self.N):
            for j in range(self.M):
                bij_low[i][j] = cal.cal_membership(y=self.miu1[j], miu=self.miu3[i], sigma=self.sigma3_low[i])
                bij_high[i][j] = cal.cal_membership(y=self.miu1[j], miu=self.miu3[i], sigma=self.sigma3_high[i])
            if bij_low[i, :].sum() != 0:
                self.bij_low[i, :] = bij_low[i, :] / bij_low[i, :].sum()
            if bij_high[i, :].sum() != 0:
                self.bij_high[i, :] = bij_high[i, :] / bij_high[i, :].sum()

    def forward(self, Obs, scaling):
        T = len(Obs) 
        c_low, c_high = np.zeros([T], float), np.zeros([T], float) 
        Alpha_low, Alpha_high = np.zeros([self.N, T], float), np.zeros([self.N, T], float)
        
        for i in range(self.N):
            Alpha_low[i, 0] = self.Pi_low[i] * self.it_hat_low[i, 0]
            Alpha_high[i, 0] = self.Pi_high[i] * self.it_hat_high[i, 0]
        if scaling:
            c_low[0] = 1 / Alpha_low[:, 0].sum()
            Alpha_low[:, 0] = Alpha_low[:, 0] * c_low[0]
            c_high[0] = 1 / Alpha_high[:, 0].sum()
            Alpha_high[:, 0] = Alpha_high[:, 0] * c_high[0]

      
        for t in range(1, T):
            for i in range(self.N):
                alpha_hat_low, alpha_hat_high = 0, 0
                for k in range(self.N):
                    alpha_hat_low += Alpha_low[k, t - 1] * self.T_low[k, i]
                    alpha_hat_high += Alpha_high[k, t - 1] * self.T_high[k, i]
                Alpha_low[i, t] = alpha_hat_low * self.it_hat_low[i, t]
                Alpha_high[i, t] = alpha_hat_high * self.it_hat_high[i, t]
            if scaling:
                c_low[t] = 1 / Alpha_low[:, t].sum()
                Alpha_low[:, t] = Alpha_low[:, t] * c_low[t]
                c_high[t] = 1 / Alpha_high[:, t].sum()
                Alpha_high[:, t] = Alpha_high[:, t] * c_high[t]

        log_prob_low, log_prob_high = 0, 0
        for t in range(T):
            log_prob_low += - np.log(c_low[t])
            log_prob_high += - np.log(c_high[t])
         
        log_prob = (log_prob_low + log_prob_high) / 2
        return (log_prob, log_prob_low, log_prob_high, Alpha_low, Alpha_high)

    def backward(self, Obs, scaling):
        T = len(Obs)
        c_low, c_high = np.zeros([T], float), np.zeros([T], float)

        Beta_low, Beta_high = np.zeros([self.N, T], float), np.zeros([self.N, T], float)
        Beta_low[:, T - 1], Beta_high[:, T - 1] = 1, 1
        if scaling:
            c_low[T - 1] = 1 / Beta_low[:, T - 1].sum()
            Beta_low[:, T - 1] = Beta_low[:, T - 1] * c_low[T - 1]
            c_high[T - 1] = 1 / Beta_high[:, T - 1].sum()
            Beta_high[:, T - 1] = Beta_high[:, T - 1] * c_high[T - 1]

        for t in reversed(range(T - 1)):
            for i in range(self.N):
                for k in range(self.N):
                    Beta_low[i, t] += self.T_low[i, k] * self.it_hat_low[k, t + 1] * Beta_low[k, t + 1]
                    Beta_high[i, t] += self.T_high[i, k] * self.it_hat_high[k, t + 1] * Beta_high[k, t + 1]

            if scaling:
                c_low[t] = 1 / Beta_low[:, t].sum()
                Beta_low[:, t] = Beta_low[:, t] * c_low[t]
                c_high[t] = 1 / Beta_high[:, t].sum()
                Beta_high[:, t] = Beta_high[:, t] * c_high[t]
        return Beta_low, Beta_high

    def predict_next_score(self, Obs_seq):
        pred_high, pred_low, pred = [], [], []
        self.Obtain_Eij()
        for no_Obs in range(len(Obs_seq)):
            Obs = Obs_seq[no_Obs]
            T = len(Obs)
            self.jot = np.zeros([self.M, T], float) 
            for t in range(T):
                for j in range(self.M):
                    self.jot[j, t] = cal.cal_membership(float(Obs[t]), miu=self.miu1[j], sigma=self.sigma1[j])

            self.ijt_hat_low, self.ijt_hat_high = np.zeros([self.N, self.M, T], float), np.zeros([self.N, self.M, T],
                                                                                                 float)
            self.it_hat_low, self.it_hat_high = np.zeros([self.N, T], float), np.zeros([self.N, T], float)
            for t in range(T):
                for i in range(self.N):
                    for j in range(self.M):
                        self.ijt_hat_low[i, j, t] = self.bij_low[i, j] * self.jot[j, t]
                        self.ijt_hat_high[i, j, t] = self.bij_high[i, j] * self.jot[j, t]
                        self.it_hat_low[i, t] += self.ijt_hat_low[i, j, t]
                        self.it_hat_high[i, t] += self.ijt_hat_high[i, j, t]

            log_prob, log_prob_low, log_prob_high, Alpha_low, Alpha_high = self.forward(Obs, True)
            FuzzyScore_low, FuzzyScore_high = np.zeros([self.M, T], float), np.zeros([self.M, T], float)
            score_pre_low, score_pre_high = np.zeros([T], float), np.zeros([T], float)
            for t in range(T):
                for j1 in range(self.M):
                    for i in range(self.N):
                        FuzzyScore_low[j1, t] += Alpha_low[i, t] * self.bij_low[i, j1]
                        FuzzyScore_high[j1, t] += Alpha_high[i, t] * self.bij_high[i, j1]
                FuzzyScore_low[:, t] = np.array([each / FuzzyScore_low[:, t].sum() for each in FuzzyScore_low[:, t]])
                FuzzyScore_high[:, t] = np.array([each / FuzzyScore_high[:, t].sum() for each in FuzzyScore_high[:, t]])
                for j2 in range(self.M):
                    score_pre_low[t] += FuzzyScore_low[j2, t] * self.miu1[j2]
                    score_pre_high[t] += FuzzyScore_high[j2, t] * self.miu1[j2]
                score_pre = (score_pre_low + score_pre_high) / 2
            pred_low.append(score_pre_low)
            pred_high.append(score_pre_high)
            pred.append(list(score_pre))

        return pred

    def baum_welch(self, Obs_seq, test, arg, **args):
        epochs = args['epochs'] if 'epochs' in args else 100
        epsilon = args['epsilon'] if 'epsilon' in args else 0.1
        scaling = True
        LLS, LLS_low, LLS_high = [], [], []
        tr_cc_rmse, tr_cc_mae, tr_cb_rmse, tr_cb_mae, tr_bb_rmse, tr_bb_mae, tr_bb_auc = [], [], [], [], [], [], []
        te_cc_rmse, te_cc_mae, te_cb_rmse, te_cb_mae, te_bb_rmse, te_bb_mae, te_bb_auc = [], [], [], [], [], [], []
        time1 = time.time()
        for epoch in tqdm(range(epochs)):

            print ("Epoch " + str(epoch))
            exp_Ci_t0_low, exp_Ci_t0_high = np.zeros([self.N], float), np.zeros([self.N], float) 
            exp_num_from_Ci_low, exp_num_from_Ci_high = np.zeros([self.N], float), np.zeros([self.N], float)  
            exp_num_in_Ci_low, exp_num_in_Ci_high = np.zeros([self.N], float), np.zeros([self.N], float)
            exp_num_Ci_Ck_low, exp_num_Ci_Ck_high = np.zeros([self.N * self.N], float).reshape(self.N, self.N), np.zeros([self.N * self.N], float).reshape(self.N, self.N)
            exp_num_in_Ci_Sj_low, exp_num_in_Ci_Sj_high = np.zeros([self.N, self.M], float).reshape(self.N, self.M), np.zeros([self.N, self.M], float).reshape(self.N, self.M)

            LogLikelihood, LogLikelihood_low, LogLikelihood_high = 0, 0, 0
            number_Obs = 0

            for Obs in Obs_seq:
                number_Obs += 1
                T = len(Obs)
                self.Obtain_Eij()
                self.jot = np.zeros([self.M, T], float) 
                for t in range(T):
                    for j in range(self.M):
                        self.jot[j, t] = cal.cal_membership(float(Obs[t]), miu=self.miu1[j], sigma=self.sigma1[j])

                self.ijt_hat_low, self.ijt_hat_high = np.zeros([self.N, self.M, T], float), np.zeros([self.N, self.M, T], float)
                self.it_hat_low, self.it_hat_high = np.zeros([self.N, T], float), np.zeros([self.N, T], float)
                for t in range(T):
                    for i in range(self.N):
                        for j in range(self.M):
                            self.ijt_hat_low[i, j, t] = self.bij_low[i, j] * self.jot[j, t]
                            self.ijt_hat_high[i, j, t] = self.bij_high[i, j] * self.jot[j, t]
                            self.it_hat_low[i, t] += self.ijt_hat_low[i, j, t]
                            self.it_hat_high[i, t] += self.ijt_hat_high[i, j, t]

                log_prob, log_prob_low, log_prob_high, Alpha_low, Alpha_high= self.forward(Obs, scaling)
                Beta_low, Beta_high = self.backward(Obs, scaling)
                LogLikelihood += log_prob
                LogLikelihood_low += log_prob_low
                LogLikelihood_high += log_prob_high


                raw_Gamma_low, raw_Gamma_high = Alpha_low * Beta_low, Alpha_high * Beta_high
                Gamma_low, Gamma_high = np.zeros([self.N, T], float), np.zeros([self.N, T], float)

                for t in range(T):
                    if raw_Gamma_low[:, t].sum() != 0:
                        Gamma_low[:, t] = raw_Gamma_low[:, t] / raw_Gamma_low[:, t].sum()
                    else:
                        Gamma_low[:, t] = [1 / self.N] * self.N
                    if raw_Gamma_high[:, t].sum() != 0:
                        Gamma_high[:, t] = raw_Gamma_high[:, t] / raw_Gamma_high[:, t].sum()
                    else:
                        Gamma_high[:, t] = [1 / self.N] * self.N
                    
                exp_Ci_t0_low += Gamma_low[:, 0]
                exp_Ci_t0_high += Gamma_high[:, 0]

                exp_num_from_Ci_low += Gamma_low[:, :T - 1].sum(1)
                exp_num_from_Ci_high += Gamma_high[:, :T - 1].sum(1)

                exp_num_in_Ci_low += Gamma_low.sum(1)
                exp_num_in_Ci_high += Gamma_high.sum(1)

                Gamma_star_low, Gamma_star_high = np.zeros([self.N, self.M], float), np.zeros([self.N, self.M], float)
                for i in range(self.N):
                    for j in range(self.M):
                        for t in range(T):
                            if self.it_hat_low[i, t] != 0:
                                Gamma_star_low[i, j] += Gamma_low[i, t] * (self.ijt_hat_low[i, j, t] / self.it_hat_low[i, t])
                            if self.it_hat_high[i, t] != 0:
                                Gamma_star_high[i, j] += Gamma_high[i, t] * (self.ijt_hat_high[i, j, t] / self.it_hat_high[i, t])
                exp_num_in_Ci_Sj_low += Gamma_star_low
                exp_num_in_Ci_Sj_high += Gamma_star_high

                Xi_low, Xi_high = np.zeros([self.N, self.N], float), np.zeros([self.N, self.N], float)
                Xi_hat_low, Xi_hat_high = np.zeros([T - 1, self.N, self.N], float), np.zeros([T - 1, self.N, self.N], float)
                kesi_low, kesi_high = np.zeros([T - 1], float), np.zeros([T - 1], float)
                for t in range(T - 1):
                    for i in range(self.N):
                        for k in range(self.N):
                            Xi_hat_low[t, i, k] = Alpha_low[i, t] * self.T_low[i, k] * self.it_hat_low[k, t + 1] * Beta_low[k, t + 1]
                            Xi_hat_high[t, i, k] = Alpha_high[i, t] * self.T_high[i, k] * self.it_hat_high[k, t + 1] * Beta_high[k, t + 1]
                            kesi_low[t] += Xi_hat_low[t, i, k]
                            kesi_high[t] += Xi_hat_high[t, i, k]
                    Xi_low += Xi_hat_low[t, :, :] / kesi_low[t]
                    Xi_high += Xi_hat_high[t, :, :] / kesi_high[t]
                exp_num_Ci_Ck_low += Xi_low
                exp_num_Ci_Ck_high += Xi_high


            self.Pi_low = exp_Ci_t0_low / exp_Ci_t0_low.sum()
            self.Pi_high = exp_Ci_t0_high / exp_Ci_t0_high.sum()

            T_hat_low, T_hat_high, T_low, T_high = np.zeros([self.N, self.N], float).reshape(self.N, self.N),\
                                                   np.zeros([self.N, self.N], float).reshape(self.N, self.N), \
                                                   np.zeros([self.N, self.N], float).reshape(self.N, self.N),\
                                                   np.zeros([self.N, self.N], float).reshape(self.N, self.N)
            for i in range(self.N):
                T_hat_low[i, :] = exp_num_Ci_Ck_low[i, :] / exp_num_from_Ci_low[i]
                T_low[i, :] = T_hat_low[i, :] / T_hat_low[i, :].sum()
                T_hat_high[i, :] = exp_num_Ci_Ck_high[i, :] / exp_num_from_Ci_high[i]
                T_high[i, :] = T_hat_high[i, :] / T_hat_high[i, :].sum()
            self.T_low, self.T_high = T_low, T_high

            miu3_low, miu3_high = np.zeros(self.N, float), np.zeros(self.N, float)
            sigma3_low, sigma3_high = np.zeros(self.N, float), np.zeros(self.N, float)
            Ci_Sj_low, Ci_Sj_high = np.zeros([self.N, self.M], float).reshape(self.N, self.M), \
                                    np.zeros([self.N, self.M], float).reshape(self.N, self.M)

            for i in range(self.N):
                Ci_Sj_low[i, :] = exp_num_in_Ci_Sj_low[i, :] / exp_num_in_Ci_low[i]
                Ci_Sj_high[i, :] = exp_num_in_Ci_Sj_high[i, :] / exp_num_in_Ci_high[i]
                Ci_Sj_low[i, :] /= Ci_Sj_low[i, :].sum()
                Ci_Sj_high[i, :] /= Ci_Sj_high[i, :].sum()
                for j in range(self.M):
                    miu3_low[i] += self.miu1[j] * Ci_Sj_low[i, j]
                    miu3_high[i] += self.miu1[j] * Ci_Sj_high[i, j]
                self.miu3[i] = (miu3_low[i] + miu3_high[i]) / 2
                for j in range(self.M):
                    sigma3_low[i] += ((self.miu1[j] - self.miu3[i]) ** 2 * Ci_Sj_low[i, j])
                    sigma3_high[i] += ((self.miu1[j] - self.miu3[i]) ** 2 * Ci_Sj_high[i, j])
                if (sigma3_low[i] != 0):
                    self.sigma3_low[i] = np.sqrt(sigma3_low[i])
                if (sigma3_high[i] != 0):
                    self.sigma3_high[i] = np.sqrt(sigma3_high[i])

            LLS.append(LogLikelihood)
            LLS_low.append(LogLikelihood_low)
            LLS_high.append(LogLikelihood_high)
           
            c_tr_pred = self.predict_next_score(Obs_seq)
            c_te_pred = self.predict_next_score(test)

            c_tr_actual = Obs_seq
            c_te_actual = test

            b_tr_actual = []
            for i in range(len(c_tr_actual)):
                x = []
                for j in range(len(c_tr_actual[i])):
                    x.append(np.greater_equal(float(c_tr_actual[i][j]), 0.5).astype(int))
                b_tr_actual.append(x)


            b_te_actual = []
            for i in range(len(c_te_actual)):
                x = []
                for j in range(len(c_te_actual[i])):
                    x.append(np.greater_equal(float(c_te_actual[i][j]), 0.5).astype(int))
                b_te_actual.append(x)


            b_tr_pred = []
            for i in range(len(c_tr_pred)):
                x = []
                for j in range(len(c_tr_pred[i])):
                    x.append(np.greater_equal(float(c_tr_pred[i][j]), 0.5).astype(int))
                b_tr_pred.append(x)


            b_te_pred = []
            for i in range(len(c_te_pred)):
                x = []
                for j in range(len(c_te_pred[i])):
                    x.append(np.greater_equal(float(c_te_pred[i][j]), 0.5).astype(int))
                b_te_pred.append(x)

            c_tr_actual = list(chain.from_iterable(c_tr_actual))
            c_tr_pred = list(chain.from_iterable(c_tr_pred))
            b_tr_actual = list(chain.from_iterable(b_tr_actual))
            b_tr_pred = list(chain.from_iterable(b_tr_pred))

            c_te_actual = list(chain.from_iterable(c_te_actual))
            c_te_pred = list(chain.from_iterable(c_te_pred))
            b_te_actual = list(chain.from_iterable(b_te_actual))
            b_te_pred = list(chain.from_iterable(b_te_pred))

            cc_rmse, cc_mae, cb_rmse, cb_mae, bb_rmse, bb_mae, bb_auc = obtain_metrics(c_actual=c_tr_actual,
                                                                                       c_pred=c_tr_pred,
                                                                                       b_actual=b_tr_actual,
                                                                                       b_pred=b_tr_pred)
            tr_cc_rmse.append(cc_rmse), tr_cc_mae.append(cc_mae), tr_cb_rmse.append(cb_rmse), tr_cb_mae.append(cb_mae), \
            tr_bb_rmse.append(bb_rmse), tr_bb_mae.append(bb_mae), tr_bb_auc.append(bb_auc)

            print('training rmse:' + str(cc_rmse))
            print('training mae:' + str(cc_mae))


            cc_rmse, cc_mae, cb_rmse, cb_mae, bb_rmse, bb_mae, bb_auc = obtain_metrics(
                c_actual=c_te_actual,
                c_pred=c_te_pred,
                b_actual=b_te_actual,
                b_pred=b_te_pred)

            te_cc_rmse.append(cc_rmse), te_cc_mae.append(cc_mae), te_cb_rmse.append(cb_rmse), te_cb_mae.append(
                cb_mae), te_bb_rmse.append(bb_rmse), te_bb_mae.append(bb_mae), te_bb_auc.append(bb_auc)

            print('testing rmse:' + str(cc_rmse))
            print('testing mae:' + str(cc_mae))
            
        paraname = str(1 - self.k[0])
        base_dir = 'result/'+str(arg[2][1])+'S'+str(arg[2][0])+'C_forget/'

        write_matrics(tr_cc_rmse, tr_cc_mae, tr_cb_rmse, tr_cb_mae, tr_bb_rmse, tr_bb_mae, tr_bb_auc,
                      te_cc_rmse, te_cc_mae, te_cb_rmse, te_cb_mae, te_bb_rmse, te_bb_mae, te_bb_auc,
                      base_dir, arg)