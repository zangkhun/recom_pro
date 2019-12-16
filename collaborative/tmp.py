import pandas as pd

from surprise import KNNBaseline
from surprise import Reader
from surprise import Dataset


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


