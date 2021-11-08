# -*- coding:UTF-8 -*-
"""

@File:trees.py
@Description:描述
@Author:zxh
@Date:2021/11/08}

"""
from math import log

"""
   函数说明：创建测试数据
   return： dataSet(数据集）
            labels(分类属性）
""",


def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['不放贷', '放贷']
    return dataSet, labels


"""
   函数说明:计算给定数据集的经验熵（香农熵）
   Parameters:
         dataSet:数据集
   returns:
           shannonEnt 经验熵（香农熵）
"""


def calcShannonEnt(dataSet):
    # 返回数据集的行数
    numEntires = len(dataSet)
    # 保存每个标签在字典中出现的次数
    labelCounts = {}
    for featVec in dataSet:
        # 提取标签信息
        currentLabel = featVec[-1]
        # 如果标签没有放入统计次数的字典中，添加进去
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        # 标签计数
        labelCounts[currentLabel] += 1
    # 经验熵
    shannonEnt = 0.0
    for key in labelCounts:
        # 选择该标签的概率
        prob = float(labelCounts[key]) / numEntires
        # 利用公式计算
        shannonEnt -= prob * log(prob, 2)
        # 返回经验熵
    return shannonEnt


"""
   函数说明:按照给定特征划分数据集
   Parameters:
              dataSet -待划分的数据集
              axis-划分数据集的特征
              value:需要返回的特征值
"""


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis + 1])
            retDataSet.append(reduceFeatVec)
    return retDataSet


if __name__ == '__main__':
    dataSet, features = createDataSet()
    print(dataSet)
    print(calcShannonEnt(dataSet))
