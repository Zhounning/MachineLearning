import scipy.io as scio
from sklearn import decomposition
import sklearn.discriminant_analysis
from arithmetic.LDA import LDA
from arithmetic.PCA import PCA
from arithmetic.SVD import SVD

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

map_color = {0: 'g', 1: 'r', 2: 'b', 3: 'c', 4: 'y', 5: 'm'}
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体 SimHei为黑体
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
def load_data():
    # dataFile = 'data/BU3D_feature.mat'
    dataFile = 'data/BU3D-bq-final.mat'

    data = scio.loadmat(dataFile)['data']
    label = data[:,-1]
    eigenvector = data[:,0:-1]
    return (eigenvector,label)



def read_iris():
    from sklearn.datasets import load_iris
    from sklearn import preprocessing
    data_set = load_iris()
    data_x = data_set.data
    label = data_set.target + 1
    return data_x,label


def plot_PCA():
    data, label = load_data()
    print(np.shape(data))

    pca = sklearn.decomposition.PCA(n_components=2)
    pcaData = PCA(data, 2)
    pypcaData = pca.fit_transform(data)

    color = list(map(lambda x: map_color[x], label))

    plt.figure()
    plt.subplot(121)
    plt.scatter(np.array(pcaData[:, 0]), np.array(pcaData[:, 1]), c=color)
    plt.title('PCA:自己的算法')

    plt.subplot(122)
    plt.scatter(np.array(pypcaData[:, 0]), np.array(pypcaData[:, 1]), c=color)
    plt.title('PCA:sklearn中的算法')

    plt.show()

def plot_LDA():
    data, label = load_data()
    print(np.shape(data))

    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=3)
    ldaData = LDA(data, label,3)
    ldaData = np.real(ldaData)
    pyldaData = lda.fit(data, label).transform(data)

    color = list(map(lambda x: map_color[x], label))

    plt.figure()
    plt.subplot(121)
    plt.scatter(np.array(ldaData[:, 0]), np.array(ldaData[:, 1]), c=color)
    plt.title('LDA:自己的算法')

    plt.subplot(122)
    plt.scatter(np.array(pyldaData[:, 0]), np.array(pyldaData[:, 1]), c=color)
    plt.title('LDA:sklearn中的算法')

    plt.show()

def plot_SVD():
    data, label = load_data()
    print(np.shape(data))

    tsvd = decomposition.TruncatedSVD(n_components=2)

    svdData = SVD(data, 2)

    pysvdData = tsvd.fit_transform(data)

    color = list(map(lambda x: map_color[x], label))

    plt.figure()
    plt.subplot(121)
    plt.scatter(np.array(svdData[:, 0]), np.array(svdData[:, 1]), c=color)
    plt.title('SVD:自己的算法')

    plt.subplot(122)
    plt.scatter(np.array(pysvdData[:, 0]), np.array(pysvdData[:, 1]), c=color)
    plt.title('SVD:sklearn中的算法')

    plt.show()




if __name__=="__main__":
    eignvector,label = load_data()
    print(np.shape(eignvector))
    print(np.shape(label))