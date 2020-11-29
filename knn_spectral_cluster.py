import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.pyplot as plt


def kernel(x1, x2, sigma=0.07):
    return np.exp(-(np.linalg.norm(x1 - x2) ** 2) / (2 * sigma ** 2))


def distance(x1, x2):
    res = np.sum((x1 - x2) ** 2)
    return res


def similarity_matrix(X):
    X = np.array(X)
    S = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            S[i][j] = S[j][i] = distance(X[i], X[j])
    return S


def myKNN(S, k, sigma):
    N = len(S)
    A = np.zeros((N, N))
    for i in range(N):
        dist_with_index = zip(S[i], range(N))
        dist_with_index = sorted(dist_with_index, key=lambda x: x[0])
        neighbours_id = [dist_with_index[m][1] for m in range(k + 1)]  # xi's k nearest neighbours
        for j in neighbours_id:  # xj is xi's neighbour
            A[i][j] = np.exp(-S[i][j] / (2 * sigma ** 2))
            A[j][i] = A[i][j]  # mutually
    return A


def affinity_matrix(X):
    A = np.zeros((len(X), len(X)))
    for i in range(len(X) - 1):
        for j in range(i + 1, len(X)):
            A[i, j] = A[j, i] = kernel(X[i], X[j])
    return A


def getD(A):
    D = np.zeros(A.shape)
    for i in range(A.shape[0]):
        D[i, i] = np.sum(A[i, :])
    return D


def getL(D, A):
    """随机游走的正规化拉普拉斯矩阵"""
    L = D - A
    L = np.dot(np.linalg.pinv(D), L)
    return L


def get_eigen(L, num_clusters):
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    best_eigenvalues = np.argsort(eigenvalues)[0:num_clusters]
    U = eigenvectors[:, best_eigenvalues]
    return U


def cluster(data, num_clusters):
    data = np.array(data)
    S = similarity_matrix(data)
    W = myKNN(S, 10, 10)
    D = getD(W)
    L = getL(D, W)
    eigenvectors = get_eigen(L, num_clusters)
    clf = KMeans(n_clusters=num_clusters)
    s = clf.fit(eigenvectors)  # 聚类
    label = s.labels_
    return label


def plotRes(data, clusterResult, clusterNum):
    """
    结果可似化
    :param data:  样本集
    :param clusterResult: 聚类结果
    :param clusterNum: 聚类个数
    :return:
    """
    n = len(data)
    scatterColors = ['black', 'blue', 'red', 'yellow', 'green', 'purple', 'orange']
    for i in range(clusterNum):
        color = scatterColors[i % len(scatterColors)]
        x1 = []
        y1 = []
        for j in range(n):
            if clusterResult[j] == i:
                x1.append(data[j, 0])
                y1.append(data[j, 1])
        plt.scatter(x1, y1, c=color, marker='+')


if __name__ == '__main__':
    # # 月牙形数据集,sigma=0.1
    # cluster_num = 2
    # data, target = datasets.make_moons()
    # label = cluster(data, cluster_num)
    # print(label)
    # plotRes(data, label, cluster_num)

    # # 圆形数据集,k要大一点在30左右
    # cluster_num = 2
    # data, target = datasets.make_circles(n_samples=1500, shuffle=True, noise=0.03, factor=0.75)
    # label = cluster(data, cluster_num)
    # print(label)
    # plotRes(data, label, cluster_num)

    # 正态数据集
    cluster_num = 5
    data, target = datasets.make_blobs(n_samples=500, n_features=2, centers=5, random_state=24)
    label = cluster(data, cluster_num)
    print(label)
    plt.subplot(121)
    plotRes(data, target, cluster_num)
    plt.subplot(122)
    plotRes(data, label, cluster_num)

    plt.show()
