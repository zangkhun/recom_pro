import os
import numpy as np
import pandas as pd

from collections import defaultdict
from surprise import Dataset, Reader, KNNBaseline


def GetInput(csv_path, sample_frac=1):
    """
    评分,用户名,评论时间,用户ID,电影名,类型
    注意：sample_frac 采样, 保证调试通过
    """
    csv_path = "../dataset/raw/user.csv"
    data = pd.read_csv(csv_path, header=0)
    data = data.sample(frac=sample_frac, random_state=42)

    return data


def BuildCollaborativeModel(baseline=KNNBaseline, isUserBased=True):
    """ 建立协同过滤模型"""
    sim_options = {
        "name": "pearson_baseline",
        "user_based": isUserBased,
    }
    algo = baseline(sim_options=sim_options)

    return algo


def RecomTopN(predictions: list, n=10):
    """
    为每个用户推荐 topN个item. predictions已保证未出现
    {u_id: [(item_id, ratings)]}
    """
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # 按得分降序排列
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


def getItemDict(UserMovieDF, pathDict, item="user"):
    """
    :param UserMovieDF: 列名为 user, movie 的 dataframe
    :param pathDict: 字典配置存于文件 TODO
    :param item:
    :return: 返回{item:index}字典
    """
    path = pathDict[item] if item in pathDict else ""
    # preset UserMovieDF col_names
    UserMovieDF.columns = ["user", "movie"]

    itemDF = pd.read_pickle(path) if os.path.exists(path) else UserMovieDF[item]
    # 保证固定数据集生成字典一致
    itemDF = itemDF.sort_values(ascending=True)
    # print(type(itemDF.unique()), itemDF.unique().shape)
    items = itemDF.unique().flat

    name_to_id, id_to_name = {}, {}
    for e in items:
        position = np.where(items == e)[0][0]
        name_to_id[e] = position
        id_to_name[position] = e

    return name_to_id, id_to_name


def main():
    csv_path = "../dataset/raw/user.csv"

    raw = GetInput(csv_path=csv_path, sample_frac=0.01)

    # user & item 字典，便于后续输入数据转换或其它查找操作
    mv_to_id, id_to_mv = getItemDict(raw[["用户名", "电影名"]], pathDict={}, item="movie")
    usr_to_id, id_to_usr = getItemDict(raw[["用户名", "电影名"]], pathDict={}, item="user")

    raw["user"] = raw["用户名"].apply(lambda x: usr_to_id[x])
    raw["movie"] = raw["电影名"].apply(lambda x: mv_to_id[x])

    df = raw[["user", "movie", "评分"]]
    data = Dataset.load_from_df(df, reader=Reader(rating_scale=(1, 5)))
    trainset = data.build_full_trainset()

    algo = BuildCollaborativeModel()
    algo.fit(trainset)

    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)

    top_n = RecomTopN(predictions, n=10)

    # print or return top_n for each user
    for uid, user_ratings in top_n.items():
        print(uid, [item_id for (item_id, _) in user_ratings])
        break

    print("recom stage is done.")
    # return top_n


# if __name__ == '__main__':
#     main()
