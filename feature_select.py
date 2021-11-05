#coding:utf-8
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from minepy import MINE
import dcor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import random
import copy


def rank_fea(dataset):
    fea_importance_index_matrix = []

    dataset = np.array(dataset)
    fea_matrix = dataset[:,:-1]
    label_matrix = dataset[:,-1]


    # variance
    var_list = []
    for i in range(len(fea_matrix[0])):
        var = np.var(fea_matrix[:,i])
        var_list.append(var)

    var_list = np.array(var_list)
    fea_index_var = np.argsort(-var_list)    #方差从大到小
    print("fea_index_var", fea_index_var[0:40])
    fea_importance_index_matrix.append(list(fea_index_var))


    # # ka fang
    # sk = SelectKBest(chi2, k='all')
    # sk.fit(np.abs(fea_matrix), label_matrix)
    #
    # fea_index_kafang = np.argsort(- sk.scores_)         #卡方从大到小 （非负）
    # print("fea_index_kafang", fea_index_kafang[0:40])


    # Pearson correlation
    pear_list = []
    for i in range(len(fea_matrix[0])):
        pear = pearsonr(fea_matrix[:,i], label_matrix)
        pear_list.append(pear[0])
    pear_list = np.array(pear_list)
    fea_index_pear = np.argsort(-pear_list)    #相关系数从大到小
    print("fea_index_pear", fea_index_pear[0:40])
    fea_importance_index_matrix.append(list(fea_index_pear))



    # mutual_info
    mu = SelectKBest(mutual_info_classif, k='all')
    mu.fit(fea_matrix, label_matrix)

    fea_index_mutual = np.argsort(-mu.scores_)         #互信息从大到小
    print("fea_index_mutual", fea_index_mutual[0:40])
    fea_importance_index_matrix.append(list(fea_index_mutual))


    # # f value
    # f_value = SelectKBest(f_classif, k='all')
    # f_value.fit(fea_matrix, label_matrix)
    #
    # fea_index_f_value = np.argsort(-f_value.scores_)         #f值从大到小
    # print("fea_index_f_value", fea_index_f_value[0:40])
    # fea_importance_index_matrix.append(list(fea_index_f_value))
    #
    #
    #
    # #MIC
    # MIC_list = []
    # mine = MINE()
    # for i in range(len(fea_matrix[0])):
    #
    #     mine.compute_score(fea_matrix[:,i], label_matrix)
    #     MIC_list.append(mine.mic())
    # # print(MIC_list)
    # MIC_list = np.array(MIC_list)
    # fea_index_mic = np.argsort(-MIC_list)    #最大互信息系数从大到小
    # print("fea_index_mic", fea_index_mic[0:40])
    # fea_importance_index_matrix.append(list(fea_index_mic))
    #
    #
    #
    # #Distance Correlation
    # dcor_list = []
    # for i in range(len(fea_matrix[0])):
    #     dcor_value = dcor.distance_correlation(fea_matrix[:,i], label_matrix)
    #     dcor_list.append(dcor_value)
    # dcor_list = np.array(dcor_list)
    # fea_index_dcor = np.argsort(-dcor_list)    #距离相关系数从大到小
    # print("fea_index_dcor", fea_index_dcor[0:40])
    # fea_importance_index_matrix.append(list(fea_index_dcor))
    #
    #
    # #kendall 相关系数
    # kendall_list = []
    # spearman_list = []
    #
    # for i in range(len(fea_matrix[0])):
    #     A1 = pd.Series(fea_matrix[:,i])
    #     B1 = pd.Series(label_matrix)
    #
    #     kendall_value = B1.corr(A1, method='kendall')
    #     spearman_value = B1.corr(A1, method='spearman')
    #
    #
    #     kendall_list.append(kendall_value)
    #     spearman_list.append(spearman_value)
    #
    #
    # kendall_list = np.array(kendall_list)
    # spearman_list = np.array(spearman_list)
    #
    # fea_index_kendall = np.argsort(-kendall_list)    #kendall相关系数从大到小
    # fea_index_spearmen = np.argsort(-spearman_list)    #spearmen相关系数从大到小
    #
    #
    # print("fea_index_kendall", fea_index_kendall[0:40])
    # print("fea_index_spearmen", fea_index_spearmen[0:40])
    # fea_importance_index_matrix.append(list(fea_index_kendall))
    # fea_importance_index_matrix.append(list(fea_index_spearmen))
    #
    #
    #
    # # #RFE
    # # svc = SVC(kernel="linear", C=1)
    # # rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
    # # rfe.fit(fea_matrix, label_matrix)
    # # ranking = rfe.ranking_.reshape(fea_matrix[0].shape)
    # # fea_index_rfe = np.argsort(ranking)    #rfe 从小到大
    # # print("fea_index_rfe", fea_index_rfe[0:40])
    # # fea_importance_index_matrix.append(list(fea_index_rfe))
    #
    #
    #
    #tree-based feature selection
    from sklearn.ensemble import RandomForestClassifier
    #clf = ExtraTreesClassifier(n_estimators=100)
    clf = RandomForestClassifier(n_estimators=100)
    clf = clf.fit(fea_matrix, label_matrix)
    tree_fea_importance_list = clf.feature_importances_
    fea_index_tree = np.argsort(-tree_fea_importance_list)    #tree_based 重要性从大到小
    print("fea_index_tree", fea_index_tree[0:40])
    fea_importance_index_matrix.append(list(fea_index_tree))

    np.savetxt("fea_importance.csv", fea_index_tree, delimiter=",", fmt="%f")
    np.savetxt("fea_importance_value.csv", tree_fea_importance_list, delimiter=",", fmt="%f")


    fea_importance_index_matrix = np.array(fea_importance_index_matrix)

    return fea_importance_index_matrix


def random_choice(fea_importance_index_matrix):
    fea_total_vote = np.zeros([1087],dtype=int)
    vote_top_five = 1000+50*22
    for column in range(20):
        for row in range(len(fea_importance_index_matrix)):
            fea_total_vote[fea_importance_index_matrix[row][column]] += vote_top_five
        vote_top_five -= 50

    vote = 1083
    for column in range(5, len(fea_importance_index_matrix[0])):
        for row in range(len(fea_importance_index_matrix)):
            fea_total_vote[fea_importance_index_matrix[row][column]] += vote
        vote -= 1
    total_vote = np.sum(fea_total_vote)
    prob = fea_total_vote/total_vote

    print("prob",prob)
    prob_fea_index = np.argsort(-prob)
    print("prob_fea_index",prob_fea_index[0:40])

    return prob



def get_fea_select_prob(fea_prob_file, dataset, n_features):
    prob = np.loadtxt(fea_prob_file)
    if np.sum(prob)!= 1:
        prob_sort_index = np.argsort(prob)
        summary = 0
        for i in range(0,len(prob)-1):
            summary += prob[prob_sort_index[i]]
        prob[prob_sort_index[len(prob)-1]] = 1- summary

    features = list()
    index_seq = range(len(dataset[0])-1)   #特征索引序列 0,1..1086

    while len(features) < n_features:
        index = np.random.choice(index_seq, p=prob.ravel())  # 往 features 添加 n_features 个特征，特征索引从 dataset 中按概率选取
        if index not in features:
            features.append(index)

    return features


"""4 class"""
# read data
import pandas as pd

file_path = "./data/7SERS.xls"
file_path2 = "./data/class1_322.xls"


def read_excel(path, path2):
    file = pd.ExcelFile(path)
    sheet_names = file.sheet_names
    # df_1 = pd.read_excel(path, str(sheet_names[0]), header=None)
    # df_2 = pd.read_excel(path, str(sheet_names[1]), header=None)
    # df_3 = pd.read_excel(path, str(sheet_names[2]), header=None)
    df_4 = pd.read_excel(path, str(sheet_names[3]), header=None)  # 95 yi gan yang xing
    df_5 = pd.read_excel(path, str(sheet_names[4]), header=None)  # 80 bai xue
    df_6 = pd.read_excel(path, str(sheet_names[5]), header=None)  # 203 jian kang
    # df_7 = pd.read_excel(path, str(sheet_names[6]), header=None)
    # data_1 = df_1.values
    # data_2 = df_2.values
    # data_3 = df_3.values
    data_4 = df_4.values
    data_5 = df_5.values
    data_6 = df_6.values
    # data_7 = df_7.values


    file2 = pd.ExcelFile(path2)
    sheet_names2 = file2.sheet_names
    df_class1_1 = pd.read_excel(path2, str(sheet_names2[0]), header=None)
    df_class1_2 = pd.read_excel(path2, str(sheet_names2[1]), header=None)
    df_class1_3 = pd.read_excel(path2, str(sheet_names2[2]), header=None)
    data_class1_1 = df_class1_1.values
    data_class1_2 = df_class1_2.values
    data_class1_3 = df_class1_3.values

    data_class1_2 = np.delete(data_class1_2, 0, axis=1)
    data_class1_3 = np.delete(data_class1_3, 0, axis=1)

    print(df_class1_1.shape)
    print(df_class1_2.shape)
    print(df_class1_3.shape)

    data_class1_1 = np.hstack((data_class1_1, data_class1_2))
    data_class1_1 = np.hstack((data_class1_1, data_class1_3))  # 322 ru xian ai
    print(data_class1_1.shape)

    print(len(data_class1_1[0]), len(data_4[0]), len(data_5[0]), len(data_6[0]))
    return data_class1_1, data_4, data_5, data_6


def X_data_process(data):
    matrix = data.T
    feature_matrix = np.delete(matrix, 0, axis=0)
    sample_num = len(feature_matrix)
    print(sample_num)
    return feature_matrix, sample_num


data_breast_health_before_raw, data_HBV_raw, data_M5_raw, data_health_raw = read_excel(file_path, file_path2)
data_breast_health_before_fea, data_breast_health_before_num = X_data_process(data_breast_health_before_raw)
data_HBV_fea, data_HBV_num = X_data_process(data_HBV_raw)
data_M5_fea, data_M5_num = X_data_process(data_M5_raw)
data_health_fea, data_health_num = X_data_process(data_health_raw)

data = np.append(data_breast_health_before_fea, data_HBV_fea, axis=0)
data = np.append(data, data_M5_fea, axis=0)
data = np.append(data, data_health_fea, axis=0)
total_num = data_breast_health_before_num + data_HBV_num + data_M5_num + data_health_num

label = np.zeros([total_num])
for i in range(total_num):
    if i < data_breast_health_before_num:
        label[i] = 0
    elif i < data_breast_health_before_num + data_HBV_num:
        label[i] = 1
    elif i < data_breast_health_before_num + data_HBV_num + data_M5_num:
        label[i] = 2
    else:
        label[i] = 3

X = data
Y = label

"""shuffle"""
length = len(Y)
index = list(range(length))
np.random.shuffle(index)
X = X[index, :]
Y = Y[index]


print(np.shape(X))
print(np.shape(Y))

Y.resize(700, 1)

dataset_matrix = np.concatenate((X, Y), axis=1)
print(np.shape(dataset_matrix))

dataset = []
for i in range(700):
    single_sample = dataset_matrix[i]
    dataset.append(list(single_sample))

print(np.shape(dataset))


fea_importance_index_matrix = rank_fea(dataset)
print(fea_importance_index_matrix.shape)

prob = random_choice(fea_importance_index_matrix)

np.savetxt("./data/feature_prob_test.csv", prob, fmt="%.8f", delimiter=",")

features = get_fea_select_prob("./data/feature_prob_test.csv", dataset, 40)

print("features_index",features)