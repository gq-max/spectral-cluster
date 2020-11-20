import numpy as np
from sklearn.cluster import KMeans
import math
import matplotlib.pyplot as plt


def load_data(filename):
    """
    载入数据
    :param filename: 文件名
    :return:
    """
    infile = open(filename, 'r')
    data, l_x, l_y = [], [], []
    for line in infile:
        words = line.split(',')  # 以逗号分开
        x1 = float(words[0])
        x2 = float(words[1])
        y1 = int(words[2][0:1])
        l_x.append([1, x1, x2])
        l_y.append([y1])
        data.append([x1, x2, y1])
    infile.close()
    l_x = np.array(l_x)
    l_y = np.array(l_y)
    data = np.array(data)
    return data, l_x, l_y


def distance(x1, x2):
    """
    获得两个样本点之间的距离
    :param x1: 样本点1
    :param x2: 样本点2
    :return:
    """
    dist = np.sqrt(np.power(x1 - x2, 2).sum())
    return dist


def get_dist_matrix(data):
    """
    获取距离矩阵
    :param data: 样本集合
    :return: 距离矩阵
    """
    n = len(data)  # 样本总数
    dist_matrix = np.zeros((n, n))  # 初始化邻接矩阵为n×n的全0矩阵
    for i in range(n):
        for j in range(i + 1, n):
            dist_matrix[i][j] = dist_matrix[j][i] = distance(data[i], data[j])
    return dist_matrix


def getW(data, k):
    """
    获的邻接矩阵 W
    :param data: 样本集合
    :param k : KNN参数
    :return: W
    """
    n = len(data)
    dist_matrix = get_dist_matrix(data)
    W = np.zeros((n, n))
    for idx, item in enumerate(dist_matrix):
        idx_array = np.argsort(item)  # 每一行距离列表进行排序,得到对应的索引列表
        W[idx][idx_array[1:k + 1]] = 1
    transpW = np.transpose(W)
    return (W + transpW) / 2


def getD(W):
    """
    获得度矩阵
    :param W: 邻接矩阵
    :return: D
    """
    D = np.diag(sum(W))
    return D


def getL(D, W):
    """
    获得拉普拉斯矩阵
    :param W: 邻接矩阵
    :param D: 度矩阵
    :return: L
    """
    return D - W


def getEigen(L, cluster_num):
    """
    获得拉普拉斯矩阵的特征矩阵
    :param L:
    :param cluter_num: 聚类数目
    :return:
    """
    eigval, eigvec = np.linalg.eig(L)
    ix = np.argsort(eigval)[0:cluster_num]
    return eigvec[:, ix]


def plotRes(data, clusterResult, clusterNum):
    """
    结果可似化
    :param data:  样本集
    :param clusterResult: 聚类结果
    :param clusterNum: 聚类个数
    :return:
    """
    n = len(data)
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange']
    for i in range(clusterNum):
        color = scatterColors[i % len(scatterColors)]
        x1 = [];
        y1 = []
        for j in range(n):
            if clusterResult[j] == i:
                x1.append(data[j, 0])
                y1.append(data[j, 1])
        plt.scatter(x1, y1, c=color, marker='+')
    plt.show()


def cluster(data, cluster_num, k):
    data = np.array(data)
    W = getW(data, k)
    D = getD(W)
    L = getL(D, W)
    eigvec = getEigen(L, cluster_num)
    clf = KMeans(n_clusters=cluster_num)
    s = clf.fit(eigvec)  # 聚类
    label = s.labels_
    return label


if __name__ == '__main__':
    cluster_num = 2
    knn_k = 5
    filename = 'ex2data2.txt'
    data, x, y = load_data(filename=filename)
    data = data[0:-1]  # 最后一列为标签列
    label = cluster(data, cluster_num, knn_k)
    plotRes(data, label, cluster_num)
