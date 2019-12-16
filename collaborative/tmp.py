import io
import pandas as pd

from surprise import KNNBaseline
from surprise import Reader
from surprise import Dataset

from .generate_matrix import getItemDict, reverseDict


def read_item_names():
    """ movie name -> id"""
    file_name = ""

    rid_to_name = {}
    name_to_rid = {}

    return rid_to_name, name_to_rid


if __name__ == '__main__':
    # 评分,用户名,评论时间,用户ID,电影名,类型
    user = pd.read_csv("../dataset/raw/user.csv", header=0)
    raw = user[["用户ID", "电影名", "评分"]]

    # reader = Reader(line_format="user movie score", sep=",")
    reader = Reader(rating_scale=(1, 5))

    data = Dataset.load_from_df(raw, reader=reader)
    trainset = data.build_full_trainset()

    sim_options = {
        "name": "pearson_baseline",
        "user_based": False,
    }

    algo = KNNBaseline(sim_options=sim_options)
    algo.fit(trainset)

    UserDict = getItemDict(user[["用户ID", "电影名"]], pathDict={}, item="user")

    rid_to_name, name_to_rid = read_item_names()

    u_name = ""
    user_id = name_to_rid[u_name]



