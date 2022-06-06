'''
此版本采用多进程方式运行bkt
'''
from multiprocessing import Pool
from FuzzyBKT import fuzzybkt
import tools
import os

__author__ = 'lf'

if __name__ == '__main__':
    dataset = ['algebra05','algebra06','bridge_algebra06','assistments09','assistments17','assistments12']
    # name_dict = {}
    # for each in dataset:
    #     file_dir = 'E:/2#Educational data mining/1#KT/1#data\lf_code/' + each + '/BKT_data/'
    #     list = os.listdir(file_dir)
    #     name_dict[each] = list
    # dataset = ['algebra05', 'algebra06']
    # # fold_list = ['fold0']
    fold_list = ['fold0', 'fold1', 'fold2', 'fold3', 'fold4']

    CS = [(6,2),(6,4),(6,6),(6,8),(6,2),(8,4),(8,6),(8,8),(8,2),(8,4),(8,6),(8,8)]

    iter_list = [(x, y, z)
                 for x in dataset
                 for y in fold_list
                 for z in CS]

    # iter_list = [
    #     ('algebra06','fold1',[0.7,1.3]),
    #     ('algebra06','fold2',[0.7,1.3])
    # ]

    # 运行
    pool = Pool(processes=40)  # 进程数
    pool.map(fuzzybkt, iter_list)