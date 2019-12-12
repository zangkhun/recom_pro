import os
import pickle
import pandas as pd
import numpy as np

from utils import printf


def reverseDict(dic):
    return dict((v, k) for k, v in dic.items())


def df2Dict(SingleDF):
    """ 单列dataframe elem => dict, {elem: index}"""
    return dict((v, k) for k, v in SingleDF.iteritems())
    # d = {}
    # for k, v in SingleDF.iteritems():
    #     d[v] = k
    # return d


def getUserDict(path, df):
    UserDF = pd.read_pickle(path) if os.path.exists(path) else df
    UserDF.sort_values(by="user", ascending=True)
    return df2Dict(UserDF)


def getMovieDict(path, df):
    UserDF = pd.read_pickle(path) if os.path.exists(path) else df
    UserDF.sort_values(by="user", ascending=True, inplace=True)
    return df2Dict(UserDF)
    pass


def getItemDict(UserMovieDF, pathDict, item="user", reverse=False):
    """ 获取{item:index}"""
    path = pathDict[item]
    # preset UserMovieDF col_names

    itemDF = pd.read_pickle(path) if os.path.exists(path) else UserMovieDF[item]
    itemDF.sort_values(by=item, ascending=True, inplace=True)

    return df2Dict(itemDF)


def getItemMatrix(inPath):

    pass


def getUserMatrix(inPath):
    pass


def getMovieMatrix(inPath):
    pass


if __name__ == '__main__':
    user = pd.read_csv("../dataset/raw/user.csv", header=0)
    us, ms = user["用户ID"].nunique(), user["电影名"].nunique()
    matrix = np.array((us, ms), dtype=float)

    pass
