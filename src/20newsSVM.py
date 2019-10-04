"""
https://scikit-learn.org/stable/modules/multiclass.html
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import svm
import pandas as pd
import numpy as np

pd.set_option('display.width', 320)
np.set_printoptions(linewidth=320)

df = pd.read_csv('../output/20newsGroup18828.csv')
df['category_id'] = df.category.factorize()[0]

# Some details about the data
# print(df.groupby(['category', 'category_id']).count().sort_values('category_id'))

train_df, test_df = train_test_split(df, test_size=0.2)
# print(train_df.groupby(['category', 'category_id']).count().sort_values('category_id'))
# print(test_df.groupby(['category', 'category_id']).count().sort_values('category_id'))

tfidf = TfidfVectorizer()
X_train_features = tfidf.fit_transform(train_df.text)

clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
clf.fit(X_train_features, train_df.category_id)
dec = clf.decision_function([[1]])
print(dec.shape[1]) # n_class * (n_class - 1) / 2
clf.decision_function_shape = 'ovr'
dec = clf.decision_function([[1]])
print(dec.shape[1])
