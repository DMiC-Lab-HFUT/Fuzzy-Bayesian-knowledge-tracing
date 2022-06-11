
from multiprocessing import Pool
from FuzzyBKT import fuzzybkt
import tools
import os

__author__ = 'lf'

if __name__ == '__main__':
    dataset = ['algebra05','algebra06','bridge_algebra06','assistments09','assistments17','assistments12']

    fold_list = ['fold0', 'fold1', 'fold2', 'fold3', 'fold4']

    CS = [(6,2),(6,4),(6,6),(6,8),(6,2),(8,4),(8,6),(8,8),(8,2),(8,4),(8,6),(8,8)]

    iter_list = [(x, y, z)
                 for x in dataset
                 for y in fold_list
                 for z in CS]


    pool = Pool(processes=40) 
    pool.map(fuzzybkt, iter_list)