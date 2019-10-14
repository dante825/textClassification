"""
Move the TF-IDF part out of every script so it won't repeat in every script
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

pd.set_option('display.width', 320)
np.set_printoptions(linewidth=320)


def generate_tfidf():
    df = pd.read_csv('../output/20newsGroup18828.csv')
    df['category_id'] = df.category.factorize()[0]

    # Some details about the data
    # print(df.groupby(['category', 'category_id']).count().sort_values('category_id'))

    train_df, test_df = train_test_split(df, test_size=0.2)
    # print(train_df.groupby(['category', 'category_id']).count().sort_values('category_id'))
    # print(test_df.groupby(['category', 'category_id']).count().sort_values('category_id'))

    tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
    x_train_features = tfidf.fit_transform(train_df.text)
    x_test_features = tfidf.transform(test_df.text)
    # train_df['features'] = x_train_features
    # test_df['features'] = x_test_features

    return train_df, test_df, x_train_features, x_test_features


def tfidf_generator():
    # https: // www.kaggle.com / dex314 / tfidf - truncatedsvd - and -light - gbm
    df = pd.read_csv('../output/20newsGroup18828.csv')
