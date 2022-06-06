import numpy as np
from numpy import random as rand
import calculate as cal
import time
from sklearn import metrics
import tools
from obtain_metrics import *
from itertools import chain
from tqdm import tqdm
# from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score, f1_score,auc

class fuzzy_hmm:
    def __init__(self, n_state, forget_param, centers, sigma1, dict,para):
        #认知模糊集参数
        self.N = n_state #认知模糊集个数
        self.miu2 = np.array(range(self.N)).astype(float) / (self.N - 1) #认知模糊集miu

        self.sigma2_low, self.sigma2_high = np.ones(self.N).astype(float) / (self.N), \
                                            np.ones(self.N).astype(float) / (self.N) #认知模糊集sigma
        # raw_Pi = np.array([1] * self.N)
        # self.Pi_low, self.Pi_high = raw_Pi.astype(float) / raw_Pi.sum(), raw_Pi.astype(float) / raw_Pi.sum()
        # Pi = rand.uniform(0, 1, self.N).reshape(self.N)
        raw_Pi = np.array([1] * self.N)
        Pi = raw_Pi.astype(float) / raw_Pi.sum()
        self.Pi_low, self.Pi_high = Pi, Pi
        # self.rou_low, self.rou_high = np.zeros([11, self.N], float), np.zeros([11, self.N], float)

        # 得分模糊集参数
        self.miu1 = [each[0] for each in centers] #得分模糊集miu
        self.M = len(centers) #得分模糊集个数
        self.sigma1 = sigma1 #得分模糊集sigma
        # self.sigma1 = [0.5, 0.5, 0.5, 0.5]
        self.dict1 = dict #观测得分对应M个得分模糊集的隶属度（dict）

        #混淆概率参数
        #混淆概率：认知模糊集Ci至得分模糊集Sj的概率。采用二型模糊GMM函数，包含权值参数w，miu，sigma_low，sigma_high
        self.w = [each / sum(np.ones(self.M)) for each in np.ones(self.M)] #GMM的权值参数w，取均值
        self.miu3 = np.array(range(self.N)).astype(float) / (self.N - 1)
        # self.miu3 = rand.uniform(0, 1, self.N).reshape(self.N)
        self.miu3 = sorted(self.miu3)
        self.sigma3 = np.ones(self.N)
        # self.sigma3 = rand.uniform(0, 2, self.N).reshape(self.N)

        self.k = para
        self.sigma3_low = self.sigma3 * self.k[0]
        self.sigma3_high = self.sigma3 * self.k[1]
        self.bij_low, self.bij_high = np.zeros([self.N, self.M], float), np.zeros([self.N, self.M], float)

        #转移概率T
        #转移概率：认知模糊集Ci至认知模糊集Ck的概率。采用概率矩阵（N*N）表示。
        raw_T = rand.uniform(0, 1, self.N * self.N).reshape(self.N, self.N)
        ##设置不能遗忘，因此Ci向比Ci级别低的认知模糊集转换的概率为0
        if forget_param == 0:
            raw_T = np.triu(raw_T) #np.triu()返回矩阵的上三角矩阵
        self.T_low, self.T_high = raw_T.astype(float) / raw_T.sum(axis=1, keepdims=True), \
                                  raw_T.astype(float) / raw_T.sum(axis=1, keepdims=True)
        # axis = 1 以竖轴为基准，同行相加

    #计算混淆概率：认知模糊集Ci至得分模糊集Sj的概率
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
        """
        Calculate the probability of an observation sequence given the model parameters P(Obs|hmm)
        Alpha is defined as P(O_1:T,S_T|hmm)
        """
        T = len(Obs) #总时间步
        c_low, c_high = np.zeros([T], float), np.zeros([T], float) #用以计算log_prob

        # Initialization
        # x1 = np.random.randint(0, 11) #假设第一个时间步学生的认知状态为x1(实际是x1/10，此处为与rou列表的索引一致)
        Alpha_low, Alpha_high = np.zeros([self.N, T], float), np.zeros([self.N, T], float)
        # 计算第1时刻的Alpha; 对于每一个认知模糊集，每一时刻的Alpha等于该认知模糊集下得到所有观测模糊集的和
        for i in range(self.N):
            Alpha_low[i, 0] = self.Pi_low[i] * self.it_hat_low[i, 0]
            Alpha_high[i, 0] = self.Pi_high[i] * self.it_hat_high[i, 0]
        if scaling:
            c_low[0] = 1 / Alpha_low[:, 0].sum()
            Alpha_low[:, 0] = Alpha_low[:, 0] * c_low[0]
            c_high[0] = 1 / Alpha_high[:, 0].sum()
            Alpha_high[:, 0] = Alpha_high[:, 0] * c_high[0]

        # Induction
        for t in range(1, T):
            for i in range(self.N):
                # 计算t-1时刻的每种认知模糊集至t时刻的ci模糊集的概率
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

        # Termination
        log_prob_low, log_prob_high = 0, 0
        for t in range(T):
            log_prob_low += - np.log(c_low[t])
            log_prob_high += - np.log(c_high[t])
            # if c_low[t] != 0:
            #     log_prob_low += - np.log(c_low[t])
            # if c_high[t] != 0:
            #     log_prob_high += - np.log(c_high[t])
            # log_prob_low = np.log(np.sum(Alpha_low[:, T - 1]))
            # log_prob_high = np.log(np.sum(Alpha_high[:, T - 1]))
        log_prob = (log_prob_low + log_prob_high) / 2
        # print(log_prob)
        return (log_prob, log_prob_low, log_prob_high, Alpha_low, Alpha_high)

    def backward(self, Obs, scaling):
        """
        Calculate the probability of a partial observation sequence from t+1 to T given the model params.
        Beta is defined as P(O_1:T|S_T, hmm)
        """

        T = len(Obs)
        c_low, c_high = np.zeros([T], float), np.zeros([T], float)

        # Initialization
        Beta_low, Beta_high = np.zeros([self.N, T], float), np.zeros([self.N, T], float)
        Beta_low[:, T - 1], Beta_high[:, T - 1] = 1, 1
        if scaling:
            # 归一化Beta_low[:, T - 1]，Beta_high[:, T - 1]，使得每列和为1
            c_low[T - 1] = 1 / Beta_low[:, T - 1].sum()
            Beta_low[:, T - 1] = Beta_low[:, T - 1] * c_low[T - 1]
            c_high[T - 1] = 1 / Beta_high[:, T - 1].sum()
            Beta_high[:, T - 1] = Beta_high[:, T - 1] * c_high[T - 1]

        # Induction
        for t in reversed(range(T - 1)):
            #计算Ck至观测ot的发射概率
            for i in range(self.N):
                for k in range(self.N):
                    Beta_low[i, t] += self.T_low[i, k] * self.it_hat_low[k, t + 1] * Beta_low[k, t + 1]
                    Beta_high[i, t] += self.T_high[i, k] * self.it_hat_high[k, t + 1] * Beta_high[k, t + 1]

            if scaling:
                #归一化Beta_low[:, t]，Beta_high[:, t]，使得每列和为1
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
            self.jot = np.zeros([self.M, T], float)  # 就散每个得分Sj至ot的概率
            for t in range(T):
                for j in range(self.M):
                    self.jot[j, t] = cal.cal_membership(float(Obs[t]), miu=self.miu1[j], sigma=self.sigma1[j])

            # 计算该Obs下每个时刻，ci至ot的概率
            self.ijt_hat_low, self.ijt_hat_high = np.zeros([self.N, self.M, T], float), np.zeros([self.N, self.M, T],
                                                                                                 float)
            self.it_hat_low, self.it_hat_high = np.zeros([self.N, T], float), np.zeros([self.N, T], float)
            for t in range(T):
                for i in range(self.N):
                    for j in range(self.M):
                        # 从Ci经过Sj至ot的概率为
                        self.ijt_hat_low[i, j, t] = self.bij_low[i, j] * self.jot[j, t]
                        self.ijt_hat_high[i, j, t] = self.bij_high[i, j] * self.jot[j, t]
                        # 从Ci经过所有S至ot的概率为
                        self.it_hat_low[i, t] += self.ijt_hat_low[i, j, t]
                        self.it_hat_high[i, t] += self.ijt_hat_high[i, j, t]

            log_prob, log_prob_low, log_prob_high, Alpha_low, Alpha_high = self.forward(Obs, True)
            # 求每个时间步所处Sj得分模糊集的概率
            # 求每个时间步的预测得分
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

        # # 处理Obs_seq
        # true = []
        # for obs in Obs_seq:
        #     obs = [float(each) for each in obs]
        #     true.append(obs)
        # # 处理true和pred成一行
        # actual, prediction = [], []
        # for tr in true:
        #     for tr0 in tr:
        #         actual.append(tr0)
        # for pr in pred:
        #     for pr0 in pr:
        #         prediction.append(pr0)
        # MSE, MAE, Variance = tools.error_cal(actual, prediction)
        # RMSE = np.sqrt(MSE)
        # Sigma = np.sqrt(Variance)
        # # mse=metrics.mean_squared_error(true, pred)
        # R_square = tools.Rsquare(actual, prediction)
        # return MSE, RMSE, MAE, Variance, Sigma, R_square
        return pred

    def baum_welch(self, Obs_seq, test, arg, **args):
        """
        Adjust the model parameters to maximize the probability of the observation sequence given the model
        Gamma_t(i) = P(O_1:T, q_t = S_i | hmm) as the probability of in state i at time t and having the observation sequence.
        Xi_t(i,j) = P(O_1:T, q_t-1 = S_i, q_t = S_j | hmm) as the probability of transiting from state i to state j and having the observation sequence.
        """
        epochs = args['epochs'] if 'epochs' in args else 100
        epsilon = args['epsilon'] if 'epsilon' in args else 0.1
        scaling = True
        LLS, LLS_low, LLS_high = [], [], []
        tr_cc_rmse, tr_cc_mae, tr_cb_rmse, tr_cb_mae, tr_bb_rmse, tr_bb_mae, tr_bb_auc = [], [], [], [], [], [], []
        te_cc_rmse, te_cc_mae, te_cb_rmse, te_cb_mae, te_bb_rmse, te_bb_mae, te_bb_auc = [], [], [], [], [], [], []
        time1 = time.time()
        for epoch in tqdm(range(epochs)):

            print ("Epoch " + str(epoch))
            exp_Ci_t0_low, exp_Ci_t0_high = np.zeros([self.N], float), np.zeros([self.N], float) # Expected number of probability of starting from Si
            exp_num_from_Ci_low, exp_num_from_Ci_high = np.zeros([self.N], float), np.zeros([self.N], float)  # Expected number of transition from Si
            exp_num_in_Ci_low, exp_num_in_Ci_high = np.zeros([self.N], float), np.zeros([self.N], float) # Expected number of being in Si
            # Expected number of transition from Si to Sj
            exp_num_Ci_Ck_low, exp_num_Ci_Ck_high = np.zeros([self.N * self.N], float).reshape(self.N, self.N), np.zeros([self.N * self.N], float).reshape(self.N, self.N)
            # Expected number of in Si observing fuzzy sets Sj
            exp_num_in_Ci_Sj_low, exp_num_in_Ci_Sj_high = np.zeros([self.N, self.M], float).reshape(self.N, self.M), np.zeros([self.N, self.M], float).reshape(self.N, self.M)

            LogLikelihood, LogLikelihood_low, LogLikelihood_high = 0, 0, 0
            number_Obs = 0

            for Obs in Obs_seq:
                # print('Obs in Obs_seq' + str(number_Obs))
                number_Obs += 1
                T = len(Obs)
                self.Obtain_Eij() # 计算每个认知模糊集至每个得分模糊集的概率 （混淆概率）
                self.jot = np.zeros([self.M, T], float) #就散每个得分Sj至ot的概率
                for t in range(T):
                    for j in range(self.M):
                        self.jot[j, t] = cal.cal_membership(float(Obs[t]), miu=self.miu1[j], sigma=self.sigma1[j])

                # 计算该Obs下每个时刻，ci至ot的概率
                self.ijt_hat_low, self.ijt_hat_high = np.zeros([self.N, self.M, T], float), np.zeros([self.N, self.M, T], float)
                self.it_hat_low, self.it_hat_high = np.zeros([self.N, T], float), np.zeros([self.N, T], float)
                for t in range(T):
                    for i in range(self.N):
                        for j in range(self.M):
                            # 从Ci经过Sj至ot的概率为
                            self.ijt_hat_low[i, j, t] = self.bij_low[i, j] * self.jot[j, t]
                            self.ijt_hat_high[i, j, t] = self.bij_high[i, j] * self.jot[j, t]
                            # 从Ci经过所有S至ot的概率为
                            self.it_hat_low[i, t] += self.ijt_hat_low[i, j, t]
                            self.it_hat_high[i, t] += self.ijt_hat_high[i, j, t]

                log_prob, log_prob_low, log_prob_high, Alpha_low, Alpha_high= self.forward(Obs, scaling)
                Beta_low, Beta_high = self.backward(Obs, scaling)
                LogLikelihood += log_prob
                LogLikelihood_low += log_prob_low
                LogLikelihood_high += log_prob_high
                # print("Epoch " + str(epoch) + "Log Likelihood is " + str(LogLikelihood))

                #### Calculate Gamma ####
                # Gamma is defined as the probability of being in State Si at time t, given the observation sequence O1, O2, O3, O4 ,....,Ot, and the model parameters
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
                    # Gamma_low[:, t] = raw_Gamma_low[:, t] / raw_Gamma_low[:, t].sum()
                    # Gamma_high[:, t] = raw_Gamma_high[:, t] / raw_Gamma_high[:, t].sum()

                # Expected number of Ci at timestep 0
                exp_Ci_t0_low += Gamma_low[:, 0]
                exp_Ci_t0_high += Gamma_high[:, 0]

                # Expected number of transition from Ci
                exp_num_from_Ci_low += Gamma_low[:, :T - 1].sum(1)
                exp_num_from_Ci_high += Gamma_high[:, :T - 1].sum(1)

                # Expected number of being in Ci
                exp_num_in_Ci_low += Gamma_low.sum(1)
                exp_num_in_Ci_high += Gamma_high.sum(1)

                #认知模糊集Ci至得分模糊集Sj的expected number
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

                # Xi_low与Xi_high：给定模型参数和观测序列Obs的情况下，t时刻为状态模糊集Ci、t+1时刻为状态模糊集Ck的概率
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
                # for t in range(T - 2):
                    Xi_low += Xi_hat_low[t, :, :] / kesi_low[t]
                    Xi_high += Xi_hat_high[t, :, :] / kesi_high[t]
                exp_num_Ci_Ck_low += Xi_low
                exp_num_Ci_Ck_high += Xi_high

            # print('miu3')
            # print(self.miu3)
            # print('sigma3')
            # print(self.sigma3_low)
            # print(self.sigma3_high)
            # print('bij')
            # print(self.bij_low)
            # print(self.T_low)

            #Update Pi
            self.Pi_low = exp_Ci_t0_low / exp_Ci_t0_low.sum()
            self.Pi_high = exp_Ci_t0_high / exp_Ci_t0_high.sum()

            # print('Pi')
            # print(self.Pi_high)

            #Update T
            T_hat_low, T_hat_high, T_low, T_high = np.zeros([self.N, self.N], float).reshape(self.N, self.N),\
                                                   np.zeros([self.N, self.N], float).reshape(self.N, self.N), \
                                                   np.zeros([self.N, self.N], float).reshape(self.N, self.N),\
                                                   np.zeros([self.N, self.N], float).reshape(self.N, self.N)
            for i in range(self.N):
                # if exp_num_from_Ci_low[i] != 0:
                T_hat_low[i, :] = exp_num_Ci_Ck_low[i, :] / exp_num_from_Ci_low[i]
                # if T_hat_low[i, :].sum() != 0:
                T_low[i, :] = T_hat_low[i, :] / T_hat_low[i, :].sum()
                # if exp_num_from_Ci_high[i] != 0:
                T_hat_high[i, :] = exp_num_Ci_Ck_high[i, :] / exp_num_from_Ci_high[i]
                # if T_hat_high[i, :].sum() != 0:
                T_high[i, :] = T_hat_high[i, :] / T_hat_high[i, :].sum()
            self.T_low, self.T_high = T_low, T_high
            # print('T')
            # print(self.T_low)
            # print('exp_num_Ci_Ck')
            # print(exp_num_Ci_Ck_low)
            # print('exp_num_from_Ci')
            # print(exp_num_from_Ci_low)

            #更新认知模糊集Ci至得分模糊集Sj的概率的参数miu3和sigma3
            miu3_low, miu3_high = np.zeros(self.N, float), np.zeros(self.N, float)
            sigma3_low, sigma3_high = np.zeros(self.N, float), np.zeros(self.N, float)
            Ci_Sj_low, Ci_Sj_high = np.zeros([self.N, self.M], float).reshape(self.N, self.M), \
                                    np.zeros([self.N, self.M], float).reshape(self.N, self.M)

            for i in range(self.N):
                Ci_Sj_low[i, :] = exp_num_in_Ci_Sj_low[i, :] / exp_num_in_Ci_low[i]
                Ci_Sj_high[i, :] = exp_num_in_Ci_Sj_high[i, :] / exp_num_in_Ci_high[i]
                Ci_Sj_low[i, :] /= Ci_Sj_low[i, :].sum()
                Ci_Sj_high[i, :] /= Ci_Sj_high[i, :].sum()
                #计算miu3[i]
                for j in range(self.M):
                    miu3_low[i] += self.miu1[j] * Ci_Sj_low[i, j]
                    miu3_high[i] += self.miu1[j] * Ci_Sj_high[i, j]
                self.miu3[i] = (miu3_low[i] + miu3_high[i]) / 2
                # 计算sigma3[i]
                for j in range(self.M):
                    sigma3_low[i] += ((self.miu1[j] - self.miu3[i]) ** 2 * Ci_Sj_low[i, j])
                    sigma3_high[i] += ((self.miu1[j] - self.miu3[i]) ** 2 * Ci_Sj_high[i, j])
                if (sigma3_low[i] != 0):
                    self.sigma3_low[i] = np.sqrt(sigma3_low[i])
                if (sigma3_high[i] != 0):
                    self.sigma3_high[i] = np.sqrt(sigma3_high[i])

            # print('miu3')
            # print(self.miu3)
            # print('sigma3')
            # print(self.sigma3_low)
            # print('Ci_Sj_low')
            # print(Ci_Sj_low)
            # print('exp_num_in_Ci_Sj_low')
            # print(exp_num_in_Ci_Sj_low)
            # print('exp_num_in_Ci_low')
            # print(exp_num_in_Ci_low)
            # print('LogLikelihood')
            # print(str(LogLikelihood))
            # print('T')
            # print(self.T_low)
            # print('miu3')
            # print(self.miu3)
            # print(str(LogLikelihood_low))
            # print(str(LogLikelihood_high))
            LLS.append(LogLikelihood)
            LLS_low.append(LogLikelihood_low)
            LLS_high.append(LogLikelihood_high)
            # if epoch > 1:
            #     # if LLS[epoch] < LLS[epoch - 1]:
            #     #     print('LogLikelihood down')
            #     # else:
            #     #     print('LogLikelihood up')
            #     # if (abs(LLS_low[epoch] - LLS_low[epoch - 1]) < epsilon) and (abs(LLS_high[epoch] - LLS_high[epoch] < epsilon)):
            #     if abs(LLS[epoch] - LLS[epoch - 1]) < epsilon:
            #         # print(str(LogLikelihood))
            #         # print ("The loglikelihood improvement falls below threshold, training terminates at epoch " + str(epoch) + "! ")
            #         time2 = time.time()
            #         # print(str(time2 - time1))
            #         # plt.figure()
            #         # plt.plot(LLS)
            #         # plt.title('KC'+ str(kc))
            #         # plt.show()
            #         break
            # if epoch == epochs - 1:
                # plt.figure()
                # plt.plot(LLS)
                # plt.show()
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
            # b_tr_actual = [ for each in c_tr_actual]
            # b_tr_actual = b_tr_actual.astype(int)

            b_te_actual = []
            for i in range(len(c_te_actual)):
                x = []
                for j in range(len(c_te_actual[i])):
                    x.append(np.greater_equal(float(c_te_actual[i][j]), 0.5).astype(int))
                b_te_actual.append(x)

            # b_te_actual = [np.greater_equal(each, 0.5) for each in c_te_actual]
            # b_te_actual = b_te_actual.astype(int)

            b_tr_pred = []
            for i in range(len(c_tr_pred)):
                x = []
                for j in range(len(c_tr_pred[i])):
                    x.append(np.greater_equal(float(c_tr_pred[i][j]), 0.5).astype(int))
                b_tr_pred.append(x)

            # b_tr_pred = [np.greater_equal(each, 0.5) for each in c_tr_pred]
            # b_tr_pred = b_tr_pred.astype(int)

            b_te_pred = []
            for i in range(len(c_te_pred)):
                x = []
                for j in range(len(c_te_pred[i])):
                    x.append(np.greater_equal(float(c_te_pred[i][j]), 0.5).astype(int))
                b_te_pred.append(x)

            # b_te_pred = [np.greater_equal(each, 0.5) for each in c_te_pred]
            # b_te_pred = b_te_pred.astype(int)
            # 将二维数组转换成一维数组
            c_tr_actual = list(chain.from_iterable(c_tr_actual))
            c_tr_pred = list(chain.from_iterable(c_tr_pred))
            b_tr_actual = list(chain.from_iterable(b_tr_actual))
            b_tr_pred = list(chain.from_iterable(b_tr_pred))

            c_te_actual = list(chain.from_iterable(c_te_actual))
            c_te_pred = list(chain.from_iterable(c_te_pred))
            b_te_actual = list(chain.from_iterable(b_te_actual))
            b_te_pred = list(chain.from_iterable(b_te_pred))

            # TODO:指标修订(lf)20201210
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
            # EDU服务器地址
            # base_dir = 'E:/2#Educational data mining/1#KT/2#code/l_FKT_experiments/0#result_correct/T1FBKT_4S4C_forget/'
            # 本地地址
        # base_dir = 'E:\TFS\增加实验\code_correct/0#result_correct/T2FBKT_4S4C_forget_0.9/'
        paraname = str(1 - self.k[0])
        base_dir = 'E:/2#Educational data mining/1#KT/2#code\l_FKT_experiments/result_v4/T2FBKT_'+str(arg[2][1])+'S'+str(arg[2][0])+'C_forget/'

        # base_dir = 'E:/2#Educational data mining/1#KT/1#data\lf_code/result1228/T2FBKT_4S4C_forget/'
        write_matrics(tr_cc_rmse, tr_cc_mae, tr_cb_rmse, tr_cb_mae, tr_bb_rmse, tr_bb_mae, tr_bb_auc,
                      te_cc_rmse, te_cc_mae, te_cb_rmse, te_cb_mae, te_bb_rmse, te_bb_mae, te_bb_auc,
                      base_dir, arg)