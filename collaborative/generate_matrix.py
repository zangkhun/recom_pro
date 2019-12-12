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


def getItemDict(UserMovieDF, pathDict, item="user"):
    """
    :param UserMovieDF: 列名为 user, movie 的 dataframe
    :param pathDict: 字典配置存于文件
    :param item:
    :return: 返回{item:index}字典，便于矩阵填充
    """
    path = pathDict[item]
    # preset UserMovieDF col_names
    UserMovieDF.columns = ["user", "movie"]

    itemDF = pd.read_pickle(path) if os.path.exists(path) else UserMovieDF[item]
    # 保证固定数据集生成字典一致
    itemDF.sort_values(by=item, ascending=True, inplace=True)

    return df2Dict(itemDF)


def getItemMatrix(inPath):

    pass


def getUserMatrix(inPath):
    pass


def getMovieMatrix(inPath):
    pass


if __name__ == '__main__':
    pathDict = {
        "user": "",
        "movie": "",
    }
    user = pd.read_csv("../dataset/raw/user.csv", header=0)
    us, ms = user["用户ID"].nunique(), user["电影名"].nunique()

    # 构建用户矩阵
    matrix = np.array((us, ms + 1), dtype=float)

    UserDict = getItemDict(user[["用户ID", "电影名"]], pathDict, item="user")
    MovieDict = getItemDict(user[["用户ID", "电影名"]], pathDict, item="movie")

    for row in user.iterrows():
        usr, mv, score = row["用户ID"], row["电影名"], row["评分"]
        usr_id, mv_id = UserDict[usr_id], MovieDict[mv_id]
        matrix[usr_id][mv_id+1] = score

    pass
