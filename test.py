
import os


def text_save(filename, data):#filename为写入txt文件的路径，data为要写入数据列表.
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功")



result1, result2 = [], []
result1.append(0.5)
result1.append(0.9)
result1.append(1)
result2.append(100)
result2.append(120)
result2.append(130)
folder = os.path.abspath(os.path.dirname(os.getcwd())) #获得上级目录
folder_result = folder + '/data1/result2/'+ 'predict_method=' + '0' + ', forget=' + '0'
os.mkdir(folder_result)

text_save(folder_result + '/result1.txt', result1)
text_save(folder_result + '/result2.txt', result2)
