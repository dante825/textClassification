"""
Move the TF-IDF part out of every script so it won't repeat in every script
"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF, TruncatedSVD, LatentDirichletAllocation
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.base import BaseEstimator, TransformerMixin
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

    # print(train_df.groupby(['category', 'category_id']).count().sort_values('category_id'))
    # print(test_df.groupby(['category', 'category_id']).count().sort_values('category_id'))

    tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
    x = tfidf.fit_transform(df.text)
    y = df['category_id']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # To make space in memory
    del df
    return x_train, y_train, x_test, y_test


def tfidf_generator():
    # https: // www.kaggle.com / dex314 / tfidf - truncatedsvd - and -light - gbm
    df = pd.read_csv('../output/20newsGroup18828.csv')
    label = preprocessing.LabelEncoder()
    df['category_id'] = label.fit_transform(df.category)
    df = df.drop(columns=['category'])
    count_vect = CountVectorizer(analyzer=u'char', ngram_range=(1,8), max_features=1000,
                                 strip_accents='unicode', stop_words='english', token_pattern=r'\w+')

    tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=1000, strip_accents='unicode', lowercase=True,
                            analyzer='word', token_pattern=r'\w+', use_idf=True, smooth_idf=True, sublinear_tf=True,
                            stop_words='english',)

    nmfNC = 50
    nmf = NMF(n_components=nmfNC, random_state=42, alpha=.1, l1_ratio=.5)

    ldaNT = 50
    lda = LatentDirichletAllocation(n_components=ldaNT, max_iter=10, learning_method='online', learning_offset=50.,
                                    random_state=42)

    textNC = 150
    tsvdText = TruncatedSVD(n_components=textNC, n_iter=25, random_state=42)
    tsvdCount = TruncatedSVD(n_components=textNC, n_iter=25, random_state=42)

    # count_df = count_vect.fit_transform(df.text)
    # x_df = tfidf.fit_transform(count_df)

    train_df, test_df = train_test_split(df, test_size=0.2)

    class cust_txt_col(BaseEstimator, TransformerMixin):
        def __init__(self, key):
            self.key = key

        def fit(self, x, y=None):
            print('fit...')
            return self

        def transform(self, x):
            print('transform...')
            return x[self.key].apply(str)

    class cust_regression_vals(BaseEstimator, TransformerMixin):
        def fit(self, x, y=None):
            print('fit...')
            return self

        def transform(self, x):
            print('transform and drop...')
            x = x.drop(['text'], axis=1).values
            return x

    print('Pipeline...')
    fp = pipeline.Pipeline([
        ('union', pipeline.FeatureUnion(
            #            n_jobs = -1,
            transformer_list=[
                ('standard', cust_regression_vals()),
                ('pip1', pipeline.Pipeline(
                    [('text', cust_txt_col('text')), ('counts', count_vect), ('tsvdCountText', tsvdCount)])),
                ('pip2',
                 pipeline.Pipeline([('nmf_Text', cust_txt_col('text')), ('tfidf_Text', tfidf), ('nmfText', nmf)])),
                ('pip3',
                 pipeline.Pipeline([('lda_Text', cust_txt_col('text')), ('tfidf_Text', tfidf), ('ldaText', lda)])),
                ('pip4', pipeline.Pipeline(
                    [('text', cust_txt_col('text')), ('tfidf_Text', tfidf), ('tsvdText', tsvdText)]))
            ])
         )])

    for c in train_df.columns:
        if c == 'text':
            train_df[c + '_len'] = train_df[c].map(lambda x: len(str(x)))
            train_df[c + '_words'] = train_df[c].map(lambda x: len(str(x).split(' ')))

    for c in test_df.columns:
        if c == 'newText':
            test_df[c + '_len'] = test_df[c].map(lambda x: len(str(x)))
            test_df[c + '_words'] = test_df[c].map(lambda x: len(str(x).split(' ')))

    # print(train_df.head())
    # print(test_df.head())

    train = fp.fit_transform(train_df)
    # print(train.shape)
    test = fp.fit_transform(test_df)
    # print(test.shape)
    # print(train)
    # print(test)

    return train, train_df.category_id, test, test_df.category_id


# def main():
#     tfidf_generator()
#
#
# if __name__ == '__main__':
#     main()