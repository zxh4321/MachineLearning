# -*- coding:UTF-8 -*-
"""

@File:KNN.py
@Description:描述
@Author:zxh
@Date:2021/11/06}

"""
# -*- coding:UTF-8 -*-
"""

@File:KNN.py
@Description: k-近邻算法 通过计算样本之间的距离进行分类属于监督学习
@Author:zxh
@Date:2021/11/06}

"""
import operator

import numpy as np
import pandas as pd


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


group, labels = createDataSet()


def classify0(inx, dataSet, labels, k):
    # 获取dataSet的行数
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inx, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance ** 0.5
    sortedDistances = distance.argsort()
    classCount = {}
    for i in range(k):
        votelabel = labels[sortedDistances[i]]
        classCount[votelabel] = classCount.get(votelabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]



