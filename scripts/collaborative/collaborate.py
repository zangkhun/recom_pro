"""
描述：协同过滤
作者：张坤
"""
import os
import csv
import numpy as np
import pandas as pd
import time

from collections import defaultdict
from surprise import Dataset, Reader, KNNBaseline, SVD


def GetInput(csv_path, sample_frac=1):
    """
    评分,用户名,评论时间,用户ID,电影名,类型
    注意：sample_frac 采样, 保证调试通过
    """
    csv_path = "../../dataset/raw/user.csv"
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


def getItemDictV2(path="../../dataset/train", item="user"):
    path = path + "/" + item + ".csv"
    if not os.path.exists(path):
        raise Exception("先执行 handleItemDict() 保存数据")

    name_to_id, id_to_name = {}, {}
    for line in open(path, "r", encoding="utf-8"):
        index, name = line.split(",")[:2]
        index = int(index.replace("\n", ""))
        name = name.replace("\n", "")
        name = name if item == "movie" else int(name)
        name_to_id[name] = index
        id_to_name[index] = name

    return name_to_id, id_to_name


def handleItemDict(UserMovieDF, path="../../dataset/train", item="user"):
    UserMovieDF.columns = ["user", "movie"]
    itemDF = UserMovieDF[item]
    itemDF = itemDF.sort_values(ascending=True)
    items = itemDF.unique().flat

    path = path + "/" + item + ".csv"
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            for i, e in enumerate(items):
                line = str(i) + "," + str(e)
                f.write(line + "\n")

    print("data for {} is saved..".format(item))


def compute_rating_predictions(algo, data, usercol="userID", itemcol="itemID", predcol="prediction"):
    
    pass


def compute_rating_predictions(algo, data, usercol="userID", itemcol="itemID", predcol="prediction"):
    """Computes predictions of an algorithm from Surprise on the data. Can be used for computing rating metrics like RMSE.

    Args:
        algo (surprise.prediction_algorithms.algo_base.AlgoBase): an algorithm from Surprise
        data (pd.DataFrame): the data on which to predict
        usercol (str): name of the user column
        itemcol (str): name of the item column

    Returns:
        pd.DataFrame: dataframe with usercol, itemcol, predcol
    """
    predictions = [algo.predict(row[0][0], row[0][1]) for _, row in data.ir.items()]
    predictions = pd.DataFrame(predictions)
    predictions = predictions.rename(
        index=str, columns={"uid": usercol, "iid": itemcol, "est": predcol}
    )
    return predictions.drop(["details", "r_ui"], axis="columns")


def compute_ranking_predictions(algo, data, usercol="userID", itemcol="itemID", predcol="prediction", remove_seen=False,):
    """Computes predictions of an algorithm from Surprise on all users and items in data. It can be used for computing
    ranking metrics like NDCG.

    Args:
        algo (surprise.prediction_algorithms.algo_base.AlgoBase): an algorithm from Surprise
        data (pd.DataFrame): the data from which to get the users and items
        usercol (str): name of the user column
        itemcol (str): name of the item column
        remove_seen (bool): flag to remove (user, item) pairs seen in the training data

    Returns:
        pd.DataFrame: dataframe with usercol, itemcol, predcol
    """
    user_set = set()
    item_set = set()
    for _, row in data.ir.items():
        u, m = row[0][0], row[0][1]
        user_set.add(u)
        item_set.add(m)

    preds_lst = []
    for user in user_set:
        for item in item_set:
            preds_lst.append([user, item, algo.predict(user, item).est])

    all_predictions = pd.DataFrame(data=preds_lst, columns=[usercol, itemcol, predcol])

    if remove_seen:
        tempdf = pd.concat(
            [
                data[[usercol, itemcol]],
                pd.DataFrame(data=np.ones(data.shape[0]), columns=["dummycol"], index=data.index),
            ],
            axis=1,
        )
        merged = pd.merge(tempdf, all_predictions, on=[usercol, itemcol], how="outer")
        return merged[merged["dummycol"].isnull()].drop("dummycol", axis=1)
    else:
        return all_predictions


def main():
    csv_path = "../../dataset/raw/user.csv"

    raw = GetInput(csv_path=csv_path, sample_frac=0.01)

    handleItemDict(raw[["用户ID", "电影名"]], item="movie")
    handleItemDict(raw[["用户ID", "电影名"]], item="user")

    # user & item 字典，便于后续输入数据转换或其它查找操作
    mv_to_id, id_to_mv = getItemDictV2(item="movie")
    usr_to_id, id_to_usr = getItemDictV2(item="user")

    raw["user"] = raw["用户ID"].apply(lambda x: usr_to_id[x])
    raw["movie"] = raw["电影名"].apply(lambda x: mv_to_id[x])

    df = raw[["user", "movie", "评分", "评论时间"]]
    df.columns = [["userID", "itemID", "rating", "date"]]

    train = df[df["date"] <= "2018-01-23 23:59:59"]
    test = df[df["date"] > "2018-01-23 23:59:59"]

    trainset = Dataset.load_from_df(df, reader=Reader(rating_scale=(1, 5))).build_full_trainset()

    svd = SVD(random_state=0, n_factors=200, n_epochs=30, verbose=True)

    start_time = time.time()

    svd.fit(trainset)

    train_time = time.time() - start_time
    print("Took {} seconds for training.".format(train_time))

    predictions = compute_rating_predictions(svd, train, usercol='userID', itemcol='itemID')

    all_predictions = compute_ranking_predictions(svd, trainset, usercol='userID', itemcol='itemID', remove_seen=True)

    print(predictions.head())


if __name__ == '__main__':
    # max mv_id 23031, max usr_id 13544
    # test()
    main()



