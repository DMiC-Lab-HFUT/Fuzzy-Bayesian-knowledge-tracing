import os
import pandas as pd
import json

def get_file(file_dir):
    #获取文件夹内的所有文件名
    file_names = []
    for root, dirs, files in os.walk(file_dir):
        file_names.append(files)
    return file_names

def add_score(folder_KC):
    #将每个KC文件表格添加一列‘score’，以计算0-1之间的学生得分，形成的KC文件存储于add_score文件夹中
    file_names = get_file(folder_KC)
    for i in range(len(file_names[0])):
        data = pd.read_table(folder_KC + file_names[0][i], header=0, low_memory=False)
        score = []
        for j in range(len(data)):
            if data['Corrects'][j] >= 1:
                score.append(1 - w1 * data['Hints'][j] - w2 * data['Incorrects'][j])
                # w1：Hints提示的惩罚权重因子
                # w2：错误尝试次数的惩罚权重因子
            else:
                score.append(0)
        score1 = [0.0 if each <= 0.0 else each for each in score]
        score2 = [0.0 if each < 0.5 else 1.0 for each in score]
        data['score1'] = score1 # 列score1 为学生得分（0至1）
        data['score2'] = score2 # 列score2 为学生得分（0或1）
        data.to_csv(path_or_buf = 'data/add_score/' + file_names[0][i], index=False, sep='\t')
        print('add_score KC' + str(i))

def get_data(folder_stu_dict, folder_score_KC):
    #得到实验数据集score1(0至1)、score2(0或1)
    with open(folder_stu_dict, 'r', encoding='utf-8') as f:
        stu_dict = json.load(f)
    file_names = get_file(folder_score_KC)
    for i in range(len(file_names[0])):
        print(file_names[0][i])
        # data = pd.read_table(folder_score_KC + file_names[0][i], header=0, low_memory=False)
        data = pd.read_table(folder_score_KC + file_names[0][i], header=0, engine='python')
        data_grouped = data.groupby(data['Anon Student Id'])

        stu_log1, stu_log2 = [], []
        stu_name = []
        num_stu = 0
        for names, datas in data_grouped:
            # 求names对应在stu_dict中的键
            if names in stu_dict.values():
                no_names = list(stu_dict.keys())[list(stu_dict.values()).index(names)]
            stu_name.append(no_names)
            X1 = list(datas['score1'])
            X2 = list(datas['score2'])
            stu_log1.append(X1)
            stu_log2.append(X2)
            num_stu += 1
            # print(str(num_stu))

        with open('data/score1/' + file_names[0][i], 'w', encoding='utf-8') as f:
            for list_men in stu_log1:
                for b in list_men:
                    f.write(str(b) + ',')
                f.write('\n')

        with open('data/score2/' + file_names[0][i], 'w', encoding='utf-8') as f:
            for list_men in stu_log2:
                for b in list_men:
                    f.write(str(b) + ',')
                f.write('\n')

def get_str_btw(s, f, b):
    #取字符串中两个字符之间的内容
    par = s.partition(f)
    return (par[2].partition(b))[0][:]

def get_qlg(folder_score, folder_qlg):
    #得到qlg文件 qlg1(0至1) qlg2(0或1)
    file_names = get_file(folder_score)
    for i in range(len(file_names[0])):
        with open(folder_score + file_names[0][i], 'r', encoding='utf-8') as f:
            lines = f.readlines()
        qlg = []
        for line in lines:
            ea_list = line.split(',')
            qlg.append(ea_list[-2])
        qlg_name = get_str_btw(file_names[0][i], 'KCaddInt_KC', '.txt')
        with open(folder_qlg + qlg_name + '.txt', 'w', encoding='utf-8') as f:
            for line in qlg:
                f.write(line + '\t')
        print(qlg_name)

if __name__ == '__main__':
    folder = os.path.abspath(os.path.dirname(os.getcwd()))  # 获得上级目录
    folder_KC = folder + '/algebra_2005_2006_KCaddInt_part_LargerThan5/'
    folder_stu_dict = 'data/algebra_2005_2006_stu_info.json'

    folder_score_KC = 'data/add_score/'
    folder_score1 = 'data/score1/'
    folder_score2 = 'data/score2/'
    folder_qlg1 = 'data/qlg1/'
    folder_qlg2 = 'data/qlg2/'

    w1, w2 = 0.25, 0.25  # w1：Hints提示的惩罚权重因子  w2：错误尝试次数的惩罚权重因子
    # add_score(folder_KC) #执行add_score()可生成带有score的KC文件，存放在add_score文件夹内
    # get_data(folder_stu_dict, folder_score_KC)
    # get_qlg(folder_score1, folder_qlg1)
    # get_qlg(folder_score2, folder_qlg2)

    #测试
    file_name = get_file(folder_score_KC)
    file_names1 = get_file(folder_qlg1)
    file_names2 = get_file(folder_qlg2)
    l = len(file_name[0])
    l1=len(file_names1[0])
    l2=len(file_names2[0])

    print('success')