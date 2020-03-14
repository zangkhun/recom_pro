import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from keras_preprocessing import sequence


# 处理日期：yyyy-mm-dd
def get_ymd(date):
    return date.strftime("%Y-%m-%d")


# 将用户评分大于2的标记为1，小于等于2为0
def get_ctr_label(score):
    if int(score) > 2:
        return 1
    else:
        return 0


movie_df = pd.read_csv('./dataset/dataset1/movie.csv')
# 将字段处理为英文
rename_col_movie = {'类型':'Type',
             '地区':'region',
             '特色':'trait',
             '主演':'actors',
             '导演':'director',
             '电影名':'Movie_Name_CN',
              '评分':'movie_score'     }
movie_df.rename(columns=rename_col_movie, inplace=True)
print(movie_df.head(10))
user_df = pd.read_csv('./dataset/dataset1/user.csv',parse_dates=['评论时间'])

# 将字段处理为英文
rename_col_user = {'评分':'Score',
                  '用户名':'UserName',
                  '用户ID':'UserID',
                  '评论时间':'Comment_Date',
                  '电影名':'Movie_Name_CN',
                  '类型':'type'}

user_df.rename(columns=rename_col_user, inplace=True)
user_df['Comment_Date_']=user_df['Comment_Date'].apply(get_ymd)
user_df['label'] = user_df['Score'].apply(get_ctr_label)

