import os
import pandas as pd
import json

def get_file(file_dir):
    file_names = []
    for root, dirs, files in os.walk(file_dir):
        file_names.append(files)
    return file_names

def add_score(folder_KC):
    file_names = get_file(folder_KC)
    for i in range(len(file_names[0])):
        data = pd.read_table(folder_KC + file_names[0][i], header=0, low_memory=False)
        score = []
        for j in range(len(data)):
            if data['Corrects'][j] >= 1:
                score.append(1 - w1 * data['Hints'][j] - w2 * data['Incorrects'][j])

            else:
                score.append(0)
        score1 = [0.0 if each <= 0.0 else each for each in score]
        score2 = [0.0 if each < 0.5 else 1.0 for each in score]
        data['score1'] = score1 
        data['score2'] = score2 
        data.to_csv(path_or_buf = 'data/add_score/' + file_names[0][i], index=False, sep='\t')
        print('add_score KC' + str(i))

def get_data(folder_stu_dict, folder_score_KC):
    with open(folder_stu_dict, 'r', encoding='utf-8') as f:
        stu_dict = json.load(f)
    file_names = get_file(folder_score_KC)
    for i in range(len(file_names[0])):
        print(file_names[0][i])
        data = pd.read_table(folder_score_KC + file_names[0][i], header=0, engine='python')
        data_grouped = data.groupby(data['Anon Student Id'])

        stu_log1, stu_log2 = [], []
        stu_name = []
        num_stu = 0
        for names, datas in data_grouped:
            if names in stu_dict.values():
                no_names = list(stu_dict.keys())[list(stu_dict.values()).index(names)]
            stu_name.append(no_names)
            X1 = list(datas['score1'])
            X2 = list(datas['score2'])
            stu_log1.append(X1)
            stu_log2.append(X2)
            num_stu += 1

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
    par = s.partition(f)
    return (par[2].partition(b))[0][:]

def get_qlg(folder_score, folder_qlg):
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
    folder = os.path.abspath(os.path.dirname(os.getcwd()))  
    folder_KC = folder + '/algebra_2005_2006_KCaddInt_part_LargerThan5/'
    folder_stu_dict = 'data/algebra_2005_2006_stu_info.json'

    folder_score_KC = 'data/add_score/'
    folder_score1 = 'data/score1/'
    folder_score2 = 'data/score2/'
    folder_qlg1 = 'data/qlg1/'
    folder_qlg2 = 'data/qlg2/'

    w1, w2 = 0.25, 0.25 


    file_name = get_file(folder_score_KC)
    file_names1 = get_file(folder_qlg1)
    file_names2 = get_file(folder_qlg2)
    l = len(file_name[0])
    l1=len(file_names1[0])
    l2=len(file_names2[0])

    print('success')