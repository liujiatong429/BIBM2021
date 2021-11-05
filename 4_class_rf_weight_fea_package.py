# coding:utf8

from __future__ import print_function
from random import seed, randrange
import numpy as np
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn import metrics
import matplotlib as mpl
import copy
from sklearn.model_selection import StratifiedKFold
# import random

import shap


class rf_model():
    def __init__(self):
        self.trees = list()


    # Split a dataset based on an attribute and an attribute value # 根据特征和特征值分割数据集
    def test_split(self, index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def gini_index(self, groups, class_values):    # 个人理解：计算代价，分类越准确，则 gini 越小
        gini = 0.0
        D = len(groups[0]) + len(groups[1])
        for class_value in class_values:     # class_values = [0, 1]
            for group in groups:             # groups = (left, right)
                size = len(group)
                if size == 0:
                    continue
                proportion = [row[-1] for row in group].count(class_value) / float(size)
                gini += float(size)/D * (proportion * (1.0 - proportion))    # 个人理解：计算代价，分类越准确，则 gini 越小
        return gini

    def get_fea_select_prob(self, fea_prob_file, dataset, n_features):
        prob = np.loadtxt(fea_prob_file)
        if np.sum(prob) != 1:
            prob_sort_index = np.argsort(prob)
            summary = 0
            for i in range(0, len(prob) - 1):
                summary += prob[prob_sort_index[i]]
            prob[prob_sort_index[len(prob) - 1]] = 1 - summary

        features = list()
        index_seq = range(len(dataset[0]) - 1)  # 特征索引序列 0,1..1086

        while len(features) < n_features:
            index = np.random.choice(index_seq, p=prob.ravel())  # 往 features 添加 n_features 个特征，特征索引从 dataset 中按概率选取
            if index not in features:
                features.append(index)

        return features

    # 找出分割数据集的最优特征，得到最优的特征 index，特征值 row[index]，以及分割完的数据 groups（left, right）
    def get_split(self, dataset, n_features):
        class_values = list(set(row[-1] for row in dataset))  # class_values =[0, 1]
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        # features = list()
        # while len(features) < n_features:
        #     index = randrange(len(dataset[0])-1)  # 往 features 添加 n_features 个特征（ n_feature 等于特征数的根号），特征索引从 dataset 中随机取
        #     if index not in features:
        #         features.append(index)

        features = self.get_fea_select_prob("./data/feature_prob.csv", dataset, n_features)
        print("feature_index", features)

        for index in features:  # 在 n_features 个特征中选出最优的特征索引，并没有遍历所有特征，从而保证了每课决策树的差异性
            for row in dataset:
                groups = self.test_split(index, row[index],
                                    dataset)  # groups=(left, right), row[index] 遍历每一行 index 索引下的特征值作为分类值 value, 找出最优的分类特征和特征值
                gini = self.gini_index(groups, class_values)
                # 左右两边的数量越一样，说明数据区分度不高，gini系数越大
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[
                        index], gini, groups  # 最后得到最优的分类特征 b_index,分类特征值 b_value,分类结果 b_groups。b_score 为分错的代价成本
        # print b_score
        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    def to_terminal(self, group):
        outcomes = [row[-1] for row in group]  # max() 函数中，当 key 参数不为空时，就以 key 的函数对象为判断的标准
        return max(set(outcomes), key=outcomes.count)  # 输出 group 中出现次数较多的标签

    # Create child splits for a node or make terminal  # 创建子分割器，递归分类，直到分类结束
    def split(self, node, max_depth, min_size, n_features, depth):  # max_depth = 10, min_size = 1, n_features=int(sqrt((len(dataset[0])-1)
        left, right = node['groups']
        del (node['groups'])
        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            # print("depth:", depth)

            return
            # check for max depth
        if depth >= max_depth:  # max_depth=10 表示递归十次，若分类还未结束，则选取数据中分类标签较多的作为结果，使分类提前结束，防止过拟合
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            # print("depth:", depth)

            return
            # process left child
        if len(left) <= min_size:
            node['left'] = self.to_terminal(left)

        else:
            node['left'] = self.get_split(left,
                                     n_features)  # node['left']是一个字典，形式为{'index':b_index, 'value':b_value, 'groups':b_groups}，所以node是一个多层字典
            self.split(node['left'], max_depth, min_size, n_features, depth + 1)  # 递归，depth+1计算递归层数
            # process right child
        if len(right) <= min_size:
            node['right'] = self.to_terminal(right)

        else:
            node['right'] = self.get_split(right, n_features)
            self.split(node['right'], max_depth, min_size, n_features, depth + 1)

    # Build a decision tree
    def build_tree(self, train, max_depth, min_size, n_features):
        """build_tree(创建一个决策树)
        Args:
            train           训练数据集
            max_depth       决策树深度不能太深，不然容易导致过拟合
            min_size        叶子节点的大小
            n_features      选取的特征的个数
        Returns:
            root            返回决策树
        """

        # 返回最优列和相关的信息
        root = self.get_split(train, n_features)

        # 对左右2边的数据 进行递归的调用，由于最优特征使用过，所以在后面进行使用的时候，就没有意义了
        # 例如： 性别-男女，对男使用这一特征就没任何意义了
        self.split(root, max_depth, min_size, n_features, 1)

        return root

    # Make a prediction with a decision tree
    def predict(self, node, row):  # 预测模型分类结果
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):  # isinstance 是 Python 中的一个内建函数。是用来判断一个对象是否是一个已知的类型。
                return self.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], row)
            else:
                return node['right']

    # Make a prediction with a list of bagged trees
    def bagging_predict(self, row):
        """bagging_predict(bagging预测)
        Args:
            trees           决策树的集合
            row             测试数据集的每一行数据
        Returns:
            返回随机森林中，决策树结果出现次数做大的
        """

        # 使用多个决策树trees对测试集test的第row行进行预测，再使用简单投票法判断出该行所属分类
        predictions = [self.predict(tree, row) for tree in self.trees]

        class_num1 = 0
        class_num2 = 0
        class_num3 = 0
        class_num4 = 0
        for i in range(len(predictions)):
            if predictions[i] == 0:
                class_num1 += 1
            elif predictions[i] == 1.0:
                class_num2 += 1
            elif predictions[i] == 2.0:
                class_num3 += 1
            elif predictions[i] == 3.0:
                class_num4 += 1
            else:
                print("class error")

        class_all = class_num1 + class_num2 + class_num3 + class_num4
        class_list = [class_num1 / class_all, class_num2 / class_all, class_num3 / class_all, class_num4 / class_all]

        return max(set(predictions), key=predictions.count), class_list

    # Create a random subsample from the dataset with replacement
    def subsample(self, dataset, ratio):  # 创建数据集的随机子样本
        """random_forest(评估算法性能，返回模型得分)
        Args:
            dataset         训练数据集
            ratio           训练数据集的样本比例
        Returns:
            sample          随机抽样的训练样本
        """

        sample = list()
        # 训练样本的按比例抽样。
        # round() 方法返回浮点数x的四舍五入值。
        n_sample = round(len(dataset) * ratio)
        while len(sample) < n_sample:
            # 有放回的随机采样，有一些样本被重复采样，从而在训练集中多次出现，有的则从未在训练集中出现，此则自助采样法。从而保证每棵决策树训练集的差异性
            index = randrange(len(dataset))
            sample.append(dataset[index])
        return sample

    # Random Forest Algorithm
    def random_forest(self, train, max_depth, min_size, sample_size, n_trees, n_features):
        """random_forest(评估算法性能，返回模型得分)
        Args:
            train           训练数据集
            test            测试数据集
            max_depth       决策树深度不能太深，不然容易导致过拟合
            min_size        叶子节点的大小
            sample_size     训练数据集的样本比例
            n_trees         决策树的个数
            n_features      选取的特征的个数
        Returns:
            predictions     每一行的预测结果，bagging 预测最后的分类结果
        """


        # n_trees 表示决策树的数量
        for i in range(n_trees):
            # 随机抽样的训练样本， 随机采样保证了每棵决策树训练集的差异性
            sample = self.subsample(train, sample_size)
            # 创建一个决策树
            tree = self.build_tree(sample, max_depth, min_size, n_features)
            self.trees.append(tree)

        # 每一行的预测结果，bagging 预测最后的分类结果
        return self

    def predict_prob(self, test):
        predictions = []
        predictions_prob = []
        for row in test:
            single_predict, predict_prob = self.bagging_predict(row)
            predictions.append(single_predict)
            predictions_prob.append(predict_prob)

        predictions_prob = np.array(predictions_prob)

        return predictions_prob


    def predictions(self, test):
        predictions = []
        predictions_prob = []
        for row in test:
            single_predict, predict_prob = self.bagging_predict(row)
            predictions.append(single_predict)
            predictions_prob.append(predict_prob)
        predictions = np.array(predictions)

        return predictions
        # return predictions, predictions_prob






if __name__ == '__main__':

    # 加载数据
    import numpy as np
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import f1_score
    # dataset = loadDataSet('./data/sonar-all-data.csv')
    # print(type(dataset))
    # print(np.shape(dataset))
    # print(dataset)

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

    # index = list()
    # list_index_1 = list(range(data_breast_health_before_num))
    # list_index_2 = list(range(data_breast_health_before_num, data_breast_health_before_num + data_HBV_num))
    # list_index_3 = list(range(data_breast_health_before_num + data_HBV_num, data_breast_health_before_num + data_HBV_num + data_M5_num))
    # list_index_4 = list(range(data_breast_health_before_num + data_HBV_num + data_M5_num, length))
    #
    # split_1 = int(data_breast_health_before_num/10)
    # split_2 = int(data_HBV_num / 10)
    # split_3 = int(data_M5_num / 10)
    # split_4 = int(data_health_num / 10)
    #
    # for i in range(9):
    #     index.append(list_index_1[i * split_1:(i + 1) * split_1])
    #     index.append(list_index_2[i * split_2:(i + 1) * split_2])
    #     index.append(list_index_3[i * split_3:(i + 1) * split_3])
    #     index.append(list_index_4[i * split_4:(i + 1) * split_4])
    #
    # index.append(list_index_1[9 * split_1:])
    # index.append(list_index_2[9 * split_2:])
    # index.append(list_index_3[9 * split_3:])
    # index.append(list_index_4[9 * split_4:])
    #
    # index = [i for item in index for i in item]

    print(index)

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


    def cross_validation_split(dataset, n_folds):
        """
        cross_validation_split(将数据集进行抽重抽样 n_folds 份，数据可以重复重复抽取，每一次list的元素是无重复的)
        Args:
            dataset     原始数据集
            n_folds     数据集dataset分成n_flods份
        Returns:
            dataset_split    list集合，存放的是：将数据集进行抽重抽样 n_folds 份，数据可以重复重复抽取，每一次list的元素是无重复的
        """
        dataset_split = list()
        dataset_copy = list(dataset)       # 复制一份 dataset,防止 dataset 的内容改变
        fold_size = len(dataset) / n_folds
        fold_sample_index = 0
        for i in range(n_folds):
            fold = list()                  # 每次循环 fold 清零，防止重复导入 dataset_split
            while len(fold) < fold_size:   # 这里不能用 if，if 只是在第一次判断时起作用，while 执行循环，直到条件不成立
                # 有放回的随机采样，有一些样本被重复采样，从而在训练集中多次出现，有的则从未在训练集中出现，此则自助采样法。从而保证每棵决策树训练集的差异性
                index = randrange(len(dataset_copy))
                # 将对应索引 index 的内容从 dataset_copy 中导出，并将该内容从 dataset_copy 中删除。
                # pop() 函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值。
                # fold.append(dataset_copy.pop(index))  # 无放回的方式
                fold.append(dataset_copy[index])  # 有放回的方式

                # #无放回
                # index = fold_sample_index
                # fold.append(dataset_copy[index])
                # fold_sample_index += 1

            dataset_split.append(fold)
        # 由dataset分割出的n_folds个数据构成的列表，为了用于交叉验证
        return dataset_split


    # Calculate accuracy percentage
    def accuracy_metric(actual, predicted):  # 导入实际值和预测值，计算精确度
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0



    # print dataset
    # max_depth = 30, sample_size= 0.9,  n_features = 30, Trees = 10  socre = 94

    n_folds = 10      # 分成5份数据，进行交叉验证
    max_depth = 30     # 调参（自己修改） #决策树深度不能太深，不然容易导致过拟合
    min_size = 1       # 决策树的叶子节点最少的元素数量
    sample_size = 0.9  # 做决策树时候的样本的比例
    # n_features = int((len(dataset[0])-1))
    n_features = 10    # 调参（自己修改） #准确性与多样性之间的权衡
    for n_trees in [15]:  # 理论上树是越多越好


        # 将数据集进行抽重抽样 n_folds 份，数据可以重复重复抽取，每一次 list 的元素是无重复的
        folds = cross_validation_split(dataset, n_folds)
        scores = list()

        total_actual = []
        total_predict = []

        total_predict_prob = []


        # 每次循环从 folds 从取出一个 fold 作为测试集，其余作为训练集，遍历整个 folds ，实现交叉验证
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            # 将多个 fold 列表组合成一个 train_set 列表, 类似 union all
            """
            In [20]: l1=[[1, 2, 'a'], [11, 22, 'b']]
            In [21]: l2=[[3, 4, 'c'], [33, 44, 'd']]
            In [22]: l=[]
            In [23]: l.append(l1)
            In [24]: l.append(l2)
            In [25]: l
            Out[25]: [[[1, 2, 'a'], [11, 22, 'b']], [[3, 4, 'c'], [33, 44, 'd']]]
            In [26]: sum(l, [])
            Out[26]: [[1, 2, 'a'], [11, 22, 'b'], [3, 4, 'c'], [33, 44, 'd']]
            """
            train_set = sum(train_set, [])
            test_set = list()
            # fold 表示从原始数据集 dataset 提取出来的测试集
            for row in fold:
                row_copy = list(row)
                row_copy[-1] = None
                test_set.append(row_copy)

            rf = rf_model()
            rf.random_forest(train_set, max_depth, min_size, sample_size, n_trees, n_features)

            predicted = rf.predictions(test_set)
            predictions_prob = rf.predict_prob(test_set)

            # print total fold
            total_predict_prob.append(predictions_prob)

            actual = [row[-1] for row in fold]

            # 计算随机森林的预测结果的正确率
            accuracy = accuracy_metric(actual, predicted)
            scores.append(accuracy)

            print(accuracy)



            #
            #
            #
            # #print each fold
            # actual_list = list(actual)
            # predicted_list = list(predicted)
            #
            # Mix_matrix = confusion_matrix(actual_list, predicted_list)
            # precision_class_1 = Mix_matrix[0][0] / (
            #     Mix_matrix[0][0] + Mix_matrix[1][0] + Mix_matrix[2][0] + Mix_matrix[3][0])
            # precision_class_2 = Mix_matrix[1][1] / (
            #     Mix_matrix[0][1] + Mix_matrix[1][1] + Mix_matrix[2][1] + Mix_matrix[3][1])
            # precision_class_3 = Mix_matrix[2][2] / (
            #     Mix_matrix[0][2] + Mix_matrix[1][2] + Mix_matrix[2][2] + Mix_matrix[3][2])
            # precision_class_4 = Mix_matrix[3][3] / (
            #     Mix_matrix[0][3] + Mix_matrix[1][3] + Mix_matrix[2][3] + Mix_matrix[3][3])
            #
            # recall_1 = Mix_matrix[0][0] / Mix_matrix[0].sum()
            # recall_2 = Mix_matrix[1][1] / Mix_matrix[1].sum()
            # recall_3 = Mix_matrix[2][2] / Mix_matrix[2].sum()
            # recall_4 = Mix_matrix[3][3] / Mix_matrix[3].sum()
            #
            # f1_macro = f1_score(actual_list, predicted_list, average='macro')
            # f1_micro = f1_score(actual_list, predicted_list, average='micro')
            #
            #
            #
            # actual_array = np.array(actual_list)
            # y_one_hot = label_binarize(actual_array, np.arange(4))
            #
            # total_predict_prob = np.array(predictions_prob)
            # total_predict_prob = total_predict_prob.reshape((-1, 4))
            #
            # print("y_one_hot", np.shape(y_one_hot))
            # print("predictions_prob", np.shape(total_predict_prob))
            #
            # fpr, tpr, thresholds = metrics.roc_curve(y_one_hot.ravel(), total_predict_prob.ravel())
            # print("fpr", fpr)
            # auc = metrics.auc(fpr, tpr)
            # # print("auc:", auc)
            #
            # mpl.rcParams['font.sans-serif'] = u'SimHei'
            # mpl.rcParams['axes.unicode_minus'] = False
            # # FPR axis,TPR yxis
            # plt.plot(fpr, tpr, c='r', lw=2, alpha=0.7, label=u'AUC=%.3f' % auc)
            # plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
            # plt.xlim((-0.01, 1.02))
            # plt.ylim((-0.01, 1.02))
            # plt.xticks(np.arange(0, 1.1, 0.1))
            # plt.yticks(np.arange(0, 1.1, 0.1))
            # plt.xlabel('False Positive Rate', fontsize=13)
            # plt.ylabel('True Positive Rate', fontsize=13)
            # plt.grid(b=True, ls=':')
            # plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
            # plt.title(u'ROC and AUC', fontsize=17)
            # plt.show()
            #
            # # 每一次执行本文件时都能产生同一个随机数
            # seed(1)
            # # print('random=', random())
            # print('Trees: %d, n_features:%d, max_depth:%d, sample_size:%f' % (
            # n_trees, n_features, max_depth, sample_size))
            #
            # print("Confusion_Matrix")
            # print(Mix_matrix)
            # print("precision_class_1:", precision_class_1)
            # print("precision_class_2:", precision_class_2)
            # print("precision_class_3:", precision_class_3)
            # print("precision_class_4:", precision_class_4)
            #
            # print("recall_calss1:", recall_1)
            # print("recall_calss2:", recall_2)
            # print("recall_calss3:", recall_3)
            # print("recall_calss4:", recall_4)
            #
            # print("F1_macro:", f1_macro)
            # print("F1_micro:", f1_micro)
            #
            # print("auc:", auc)




            total_actual.append(actual)
            total_predict.append(predicted)


            # kernel shap
            # test = list()
            # for row in fold:
            #     row_copy = list(row)
            #     test.append(row_copy)
            #
            # train = np.array(train_set)
            # test = np.array(test)
            #
            # x_train = train[:30,:-1]
            # # y_train = train[:, -1]
            # x_test = test[:10,:-1]
            # # y_test = test[:, -1]
            #
            #
            # import shap
            #
            # explainer = shap.KernelExplainer(rf.predict_prob, x_train)
            #
            # shap_values = explainer.shap_values(x_test)
            #
            # print("len of shap values:", len(shap_values))
            # print("shape of each element", shap_values[0].shape)
            #
            # #single prediction
            # shap.initjs()
            # shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], x_test[0,:])
            #
            # shap.initjs()
            # shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], x_test[0,:])
            #
            # shap.initjs()
            # shap.force_plot(explainer.expected_value[2], shap_values[2][0,:], x_test[0,:])
            #
            # shap.initjs()
            # shap.force_plot(explainer.expected_value[3], shap_values[3][0,:], x_test[0,:])
            #
            # #predict test set
            # shap.initjs()
            # shap.force_plot(explainer.expected_value[0], shap_values[0], x_test)
            #
            # shap.initjs()
            # shap.force_plot(explainer.expected_value[1], shap_values[1], x_test)
            #
            # shap.initjs()
            # shap.force_plot(explainer.expected_value[2], shap_values[2], x_test)
            #
            # shap.initjs()
            # shap.force_plot(explainer.expected_value[3], shap_values[3], x_test)
            #
            # #summary plot
            # shap.summary_plot(shap_values, x_test)
            #
            # shap.summary_plot(shap_values[0], x_test)
            #
            # shap.summary_plot(shap_values[1], x_test)
            #
            # shap.summary_plot(shap_values[2], x_test)
            #
            # shap.summary_plot(shap_values[3], x_test)


            # #dependence plot
            #
            # shap.dependence_plot("petal length(cm)", shap_values[0], x_test)
            #
            # shap.dependence_plot("petal length(cm)", shap_values[1], x_test)
            #
            # shap.dependence_plot("petal length(cm)", shap_values[2], x_test)
            #
            # shap.dependence_plot("petal length(cm)", shap_values[3], x_test)



        total_actual_list = [x for y in total_actual for x in y]
        total_predict_list = [x for y in total_predict for x in y]

        Mix_matrix = confusion_matrix(total_actual_list, total_predict_list)
        precision_class_1 = Mix_matrix[0][0] / (
            Mix_matrix[0][0] + Mix_matrix[1][0] + Mix_matrix[2][0] + Mix_matrix[3][0])
        precision_class_2 = Mix_matrix[1][1] / (
            Mix_matrix[0][1] + Mix_matrix[1][1] + Mix_matrix[2][1] + Mix_matrix[3][1])
        precision_class_3 = Mix_matrix[2][2] / (
            Mix_matrix[0][2] + Mix_matrix[1][2] + Mix_matrix[2][2] + Mix_matrix[3][2])
        precision_class_4 = Mix_matrix[3][3] / (
            Mix_matrix[0][3] + Mix_matrix[1][3] + Mix_matrix[2][3] + Mix_matrix[3][3])

        recall_1 = Mix_matrix[0][0] / Mix_matrix[0].sum()
        recall_2 = Mix_matrix[1][1] / Mix_matrix[1].sum()
        recall_3 = Mix_matrix[2][2] / Mix_matrix[2].sum()
        recall_4 = Mix_matrix[3][3] / Mix_matrix[3].sum()

        f1_macro = f1_score(total_actual_list, total_predict_list, average='macro')
        f1_micro = f1_score(total_actual_list, total_predict_list, average='micro')

        total_actual_array = np.array(total_actual_list)

        # print("total_actual_array", np.shape(total_actual_array))

        y_one_hot = label_binarize(total_actual_array, np.arange(4))

        total_predict_prob = np.array(total_predict_prob)
        total_predict_prob = total_predict_prob.reshape((-1, 4))

        print("y_one_hot", np.shape(y_one_hot))
        print("predictions_prob", np.shape(total_predict_prob))

        fpr, tpr, thresholds = metrics.roc_curve(y_one_hot.ravel(), total_predict_prob.ravel())
        print("fpr", fpr)
        auc = metrics.auc(fpr, tpr)
        # print("auc:", auc)

        mpl.rcParams['font.sans-serif'] = u'SimHei'
        mpl.rcParams['axes.unicode_minus'] = False
        # FPR axis,TPR yxis
        plt.plot(fpr, tpr, c='r', lw=2, alpha=0.7, label=u'AUC=%.3f' % auc)
        plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
        plt.xlim((-0.01, 1.02))
        plt.ylim((-0.01, 1.02))
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('False Positive Rate', fontsize=13)
        plt.ylabel('True Positive Rate', fontsize=13)
        plt.grid(b=True, ls=':')
        plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
        plt.title(u'ROC and AUC', fontsize=17)
        plt.show()


        # 每一次执行本文件时都能产生同一个随机数
        seed(1)
        #print('random=', random())
        print('Trees: %d, n_features:%d, max_depth:%d, sample_size:%f' % (n_trees, n_features, max_depth, sample_size))


        print("Confusion_Matrix")
        print(Mix_matrix)
        print("precision_class_1:", precision_class_1)
        print("precision_class_2:", precision_class_2)
        print("precision_class_3:", precision_class_3)
        print("precision_class_4:", precision_class_4)

        print("recall_calss1:", recall_1)
        print("recall_calss2:", recall_2)
        print("recall_calss3:", recall_3)
        print("recall_calss4:", recall_4)

        print("F1_macro:", f1_macro)
        print("F1_micro:", f1_micro)

        print("auc:", auc)


        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

