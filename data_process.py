import numpy as np
import csv
import pandas as pd


file_path = "C:\\Users\\Neo\Desktop\\tong_project\data\SERS.xlsx"

def read_excel(path):
    df_1 = pd.read_excel(path, "35个健康人血浆SERS面积归一化数据", header=None)
    df_2 = pd.read_excel(path, "49个鼻咽癌血浆SERS面积归一化数据", header=None)
    df_3 = pd.read_excel(path, "32个胃癌血浆SERS面积归一化数据", header=None)
    data_1 = df_1.values
    data_2 = df_2.values
    data_3 = df_3.values

    return data_1, data_2, data_3


def X_data_process(data):
    matrix = data.T
    feature_matrix = np.delete(matrix, 0, axis=0)
    sample_num = len(feature_matrix)
    #print(sample_num)
    return feature_matrix, sample_num


data_health_raw, data_nose_raw, data_stomach_raw = read_excel(file_path)
data_health_fea, health_num = X_data_process(data_health_raw)
data_nose_fea, nose_num = X_data_process(data_nose_raw)
data_stomach_fea, stomach_num = X_data_process(data_stomach_raw)

data = np.append(data_health_fea, data_nose_fea, axis=0)
data = np.append(data, data_stomach_fea, axis=0)

total_num = health_num + nose_num + stomach_num
label = np.zeros([total_num])
for i in range(total_num):
    if i<health_num:
        label[i] = 1
    elif i< health_num+ nose_num:
        label[i] = 2
    else:
        label[i] = 3
#print(label)




