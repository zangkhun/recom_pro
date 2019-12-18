import io
import pandas as pd

from surprise import KNNBaseline
from surprise import Reader
from surprise import Dataset

from collaborative.generate_matrix import getItemDict, reverseDict


def read_item_names():
    """ movie name -> id"""
    file_name = ""

    rid_to_name = {}
    name_to_rid = {}

    return rid_to_name, name_to_rid


if __name__ == '__main__':
    # 评分,用户名,评论时间,用户ID,电影名,类型
    user = pd.read_csv("../dataset/raw/user.csv", header=0)
    user = user.sample(frac=0.01)

    raw = user[["用户ID", "电影名", "评分"]]

    # algo.fit(trainset)

    mvname_to_rid, mvid_to_name = getItemDict(raw[["用户ID", "电影名"]], pathDict={"user": "", "movie": ""}, item="movie")

    uname_to_rid, uid_to_name = getItemDict(raw[["用户ID", "电影名"]], pathDict={"user": "", "movie": ""}, item="user")

    raw["user"] = raw["用户ID"].apply(lambda x: uname_to_rid[x])
    raw["movie"] = raw["电影名"].apply(lambda x: mvname_to_rid[x])

    input = raw[["user", "movie", "评分"]]

    # reader = Reader(line_format="user movie score", sep=",")
    reader = Reader(rating_scale=(1, 5))

    data = Dataset.load_from_df(input, reader=reader)
    trainset = data.build_full_trainset()

    sim_options = {
        "name": "pearson_baseline",
        "user_based": True,
    }

    algo = KNNBaseline(sim_options=sim_options)
    algo.fit(trainset)

    u_name = 3
    user_id = uname_to_rid[u_name]

    uname_inner_id = algo.trainset.to_inner_iid(user_id)

    uname_neighbors = algo.get_neighbors(uname_inner_id, k=10)
    uname_neighbors = (algo.trainset.to_raw_iid(inner_id) for inner_id in uname_neighbors)
    uname_neighbors = (uid_to_name[r_id] if r_id in uid_to_name else -1 for r_id in uname_neighbors)

    print('The 10 nearest neighbors of Toy Story are:')

    for u in uname_neighbors:
        print(u)

    # print(input.head())

    print("--------")

    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)
    print("predictions is :", type(predictions), len(predictions), user.shape)
    print(predictions[:3])

    # rid_to_name, name_to_rid = read_item_names()
    #
    # u_name = ""
    # user_id = name_to_rid[u_name]
    #
    # u_name_inner_id = algo.trainset.to_inner_id(user_id)



