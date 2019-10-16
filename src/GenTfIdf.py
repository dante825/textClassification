"""
Move the TF-IDF part out of every script so it won't repeat in every script
"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF, TruncatedSVD, LatentDirichletAllocation
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
import pandas as pd
import numpy as np

pd.set_option('display.width', 320)
np.set_printoptions(linewidth=320)


def generate_tfidf():
    df = pd.read_csv('../output/20newsGroup18828.csv')
    # df['category_id'] = df.category.factorize()[0]
    label = preprocessing.LabelEncoder()
    df['category_id'] = label.fit_transform(df.category)

    # Some details about the data
    # print(df.groupby(['category', 'category_id']).count().sort_values('category_id'))

    train_df, test_df = train_test_split(df, test_size=0.2)
    # print(train_df.groupby(['category', 'category_id']).count().sort_values('category_id'))
    # print(test_df.groupby(['category', 'category_id']).count().sort_values('category_id'))

    tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
    x_train = tfidf.fit_transform(train_df.text)
    x_test = tfidf.transform(test_df.text)

    return x_train, train_df.category_id, x_test, test_df.category_id


def tfidf_generator():
    # https: // www.kaggle.com / dex314 / tfidf - truncatedsvd - and -light - gbm
    df = pd.read_csv('../output/20newsGroup18828.csv')

    count_vect = CountVectorizer(analyzer=u'char', ngram_range=(1,8), max_features=1000,
                                 strip_accents='unicode', stop_words='english', token_pattern=r'\w+')

    tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=1000, strip_accents='unicode', lowercase=True,
                            analyzer='word', token_pattern=r'\w+', use_idf=True, smooth_idf=True, sublinear_tf=True,
                            stop_words='english',)

    nmfNC = 50
    nmf = NMF(n_components=nmfNC, random_state=42, alpha=.1, l1_ratio=.5)

    ldaNT = 50
    lda = LatentDirichletAllocation(n_topics=ldaNT, max_iter=10, learning_method='online', learning_offset=50.,
                                    random_state=42)

    textNC = 150
    tsvdText = TruncatedSVD(n_components=textNC, n_iter=25, random_state=42)
    tsvdCount = TruncatedSVD(n_components=textNC, n_iter=25, random_state=42)

    count_df = count_vect.fit_transform(df.text)
    x_df = tfidf.fit_transform(count_df)