# -*- coding:UTF-8 -*-
"""

@File:KNN_test02.py
@Description:打开并解析文件，对数据进行分类  1代表不喜欢  2代表魅力一般  3代表极具魅力
@Author:zxh
@Date:2021/11/07}

"""
import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

import KNN


def file2matrix(filename):
    # 打开文件
    fr = open(filename)
    # 读取文件所有内容
    arrayyOLines = fr.readlines()
    # 得到文件的行数
    numberOfLines = len(arrayyOLines)
    # 返回 numpy矩阵 解析完成数据 numberOfLines行数 3列
    returnMat = np.zeros((numberOfLines, 3))
    # 返回分类器标签向量
    classLabelVector = []
    # 行索引值
    index = 0
    for line in arrayyOLines:
        # strip(rm) 当rm为空时默认删除首尾的空格（‘\n','\r','\t',' ') 也可以指定字符
        line = line.strip()
        # 对每行数据进行分割
        listFormLine = line.split('\t')
        # 将数据的前三列取出来并放入returnMat中
        returnMat[index, :] = listFormLine[0:3]
        # 根据文本中标记的喜欢程度进行分类
        if listFormLine[-1] == 'didntLike':
            classLabelVector.append(1)
        if listFormLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        if listFormLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector


"""
函数说明 ：可视化数据
Parameters:
     datingDataMat:  特征矩阵
     datingLabels: 分类Label
"""


def showDatas(datingDataMat, datingLabels):
    # 设置汉字格式
    font = FontProperties(fname="c://windows//fonts//simsun.ttc", size=14)
    # 将fig画布分隔成1行1列，不共享X轴和Y轴，fig画布的大小为（13,8）
    # 当nrow = 2，nclos = 2是代表fig画布被分割成了四个区域axs[0][0]表示第一个区域
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8))
    numberOfLabels = len(datingLabels)
    labelsColors = []
    for i in datingLabels:
        if i == 1:
            labelsColors.append('black')
        if i == 2:
            labelsColors.append('orange')
        if i == 3:
            labelsColors.append('red')
    # 画出散点图，以datingDataMat矩阵的第一列（飞行常客例行），第二列（玩游戏）数据画三点数据，散点大小为15，透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 1], color=labelsColors, s=15, alpha=.5)
    # 设置标题 X轴的label Y轴的label
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客历程数与玩视频游戏所消耗时间占比', fontproperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客历程数', fontproperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间', fontproperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图，以datingDataMat矩阵的第一列（飞行常客例行），第三列（冰激凌）数据画三点数据，散点大小为15，透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 2], color=labelsColors, s=15, alpha=.5)
    # 设置标题 X轴的label Y轴的label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激凌公升数', fontproperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客历程数', fontproperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激凌公升数', fontproperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图，以datingDataMat矩阵的第二列（玩游戏），第三列（冰激凌）数据画三点数据，散点大小为15，透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:, 1], y=datingDataMat[:, 2], color=labelsColors, s=15, alpha=.5)
    # 设置标题 X轴的label Y轴的label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激凌公升数', fontproperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比', fontproperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激凌公升数', fontproperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')

    # 设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.', markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.', markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.', markersize=6, label='largeDose')

    # 添加图例
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])

    plt.show()


"""
   函数描述：对数据进行归一化处理
"""


def autoNorm(dataSet):
    # 获取数据的最小值
    minvals = dataSet.min(0)
    maxVals = dataSet.max(0)
    # 最大值和最小值的范围
    ranges = maxVals - minvals
    # shape(dataSet)返回dataSet的行数与列数
    normDataSet = np.zeros(np.shape(dataSet))
    # 返回dataSet行数
    m = dataSet.shape[0]
    # 原始值减去最小值
    normDataSet = dataSet - np.tile(minvals, (m, 1))
    # 除以最大值和最小值的差得到归一化数据
    normDataSet = normDataSet / (np.tile(ranges, (m, 1)))
    # 返回归一化数据结果，数据范围，最小值
    return normDataSet, ranges, minvals


"""
函数说明：分类器测试
"""


def datingClassTest():
    filename = "D://machine-learning//MachineLearning//DataSet//01 k-近邻算法//datingTestSet.txt"
    datingDataMat, datingLabels = file2matrix(filename)
    # 取所有数据的百分之十
    hoRatio = 0.10
    # 数据归一化，返回归一化后的矩阵，数据范围，最小值
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 获取normMat的函数
    m = normMat.shape[0]
    # 百分之十的测试数据的个数
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        # 前numTestVecs作为测试集，后m-numTestVecs作为训练集
        classifierResult = KNN.classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 4)
        print("分类结果：%d\t真实类别：%d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("错误率：%f%%" % (errorCount / float(numTestVecs) * 100))

"""
   函数说明：通过输入一个人的三维特质，进行分类输出
"""
def classifyPerson():
    # 输出结果
    resultList = ['讨厌', '有些喜欢', '非常喜欢']
    # 三维特征输入
    precentTats = float(input("玩视频游戏所消时间百分比："))
    ffMiles = float(input("每年获得的飞行常客里程数"))
    iceCream = float(input("每周消耗的冰激凌公斤数"))
    # 打开的文件名
    filename = "D://machine-learning//MachineLearning//DataSet//01 k-近邻算法//datingTestSet.txt"
    #打开并处理数据
    datingDataMat,datingLabels = file2matrix(filename)
    # 训练集归一化处理
    normMat,ranges,minVals = autoNorm(datingDataMat)
    #生成NUmpy数组，测试集
    inArr = np.array([ffMiles,precentTats,iceCream])
    # 测试集归一化
    norminArr = (inArr-minVals)/ranges
    # 返回分类结果
    classifierResult = KNN.classify0(norminArr,normMat,datingLabels,3)
    # 打印结果
    print("你可能%s这个人"%(resultList[classifierResult-1]))

if __name__ == '__main__':
    classifyPerson()
    # datingClassTest()
    # filename = "D://machine-learning//MachineLearning//DataSet//01 k-近邻算法//datingTestSet.txt"
    # datingDataMat, datingLabels = file2matrix(filename)
    # normDataSet,ranges,minVals = autoNorm(datingDataMat)
    # print(normDataSet)
    # print(ranges)
    # print(minVals)
    # print(datingDataMat)
    # print(datingLabels)
    # showDatas(datingDataMat, datingLabels)
