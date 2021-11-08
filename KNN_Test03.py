# -*- coding:UTF-8 -*-
"""

@File:KNN_Test03.py
@Description:将32*32图像转为1*1024
@Author:zxh
@Date:2021/11/08}

"""
import numpy as np
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as KNN


def img2vector(filename):
    # 创建1*1024矩阵
    returnVect = np.zeros((1, 1024))
    # 打开文件
    fr = open(filename)
    # 按行读取
    for i in range(32):
        # 读取一行数据
        linStr = fr.readline()
        # 每一行的前32个元素一次添加到returnVect中
        for j in range(32):
            returnVect[0, 32 * i + j] = int(linStr[j])
    # 返回转换后的1*1024向量
    return returnVect


"""
   函数说明: 手写数字分类测试
"""


def handwritingClassTest():
    trainingDigitsAddress = 'D:\\machine-learning\\MachineLearning\\DataSet\\01 k-近邻算法\\trainingDigits'
    testDigitsAddress = 'D:\\machine-learning\\MachineLearning\\DataSet\\01 k-近邻算法\\testDigits'
    # 测试集的Labels
    hwLabels = []
    # 返回trainingDigits目录下的文件名
    trainingFileList = listdir(trainingDigitsAddress)
    # 返回文件夹下的文件的个数
    m = len(trainingFileList)
    # 初始化训练的Mat矩阵，测试集
    trainingMat = np.zeros((m, 1024))
    # 从文件名中解析出训练集的类别
    for i in range(m):
        # 获得文件的名字
        fileNameStr = trainingFileList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        # 将获得的类别添加到hwLabels中
        hwLabels.append(classNumber)
        # 将每一个文件的1*1024数据存储到trainingMat矩阵中   %s 表示格式化一个对象为字符
        trainingMat[i, :] = img2vector(trainingDigitsAddress + '\\%s' % (fileNameStr))
    # 构建KNN分类器
    neigh = KNN(n_neighbors=3, algorithm='auto')
    # 拟合模型， trainingMat为训练矩阵，hwLabels为对应的标签
    neigh.fit(trainingMat, hwLabels)
    # 返回testDigits目录下的文件列表
    testFileList = listdir(testDigitsAddress)
    # 错误检测计数
    errorCount = 0.0
    # 测试数据数量
    mTest = len(testFileList)
    # 从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        # 获得文件名字
        fileNameStr = testFileList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        # 获得测试集的1*1024向量用于训练
        vectorUnderTest = img2vector(testDigitsAddress + '\\%s' % (fileNameStr))
        # 获得预测结果
        classifierResult = neigh.predict(vectorUnderTest)
        print("分类返回结果%d\t真实结果%d" % (classifierResult, classNumber))
        if (classifierResult != classNumber):
            errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount / mTest * 100))


if __name__ == '__main__':
    handwritingClassTest()
