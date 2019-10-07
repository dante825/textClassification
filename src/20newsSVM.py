"""
https://scikit-learn.org/stable/modules/multiclass.html
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import svm
from sklearn.svm import LinearSVC
import time
import pandas as pd
import numpy as np

pd.set_option('display.width', 320)
np.set_printoptions(linewidth=320)

start_time = time.time()
df = pd.read_csv('../output/20newsGroup18828.csv')
df['category_id'] = df.category.factorize()[0]

# Some details about the data
# print(df.groupby(['category', 'category_id']).count().sort_values('category_id'))

train_df, test_df = train_test_split(df, test_size=0.2)
# print(train_df.groupby(['category', 'category_id']).count().sort_values('category_id'))
# print(test_df.groupby(['category', 'category_id']).count().sort_values('category_id'))

tfidf = TfidfVectorizer()
X_train_features = tfidf.fit_transform(train_df.text)
X_test_features = tfidf.transform(test_df.text)

# clf = svm.SVC(gamma='auto', decision_function_shape='ovo')
# clf.fit(X_train_features, train_df.category_id)
# dec = clf.decision_function([[1]])
# print(dec.shape[1]) # n_class * (n_class - 1) / 2
# clf.decision_function_shape = 'ovo'
# dec = clf.decision_function([[1]])
# print(dec.shape[1])

clf = LinearSVC(random_state=0, tol=1e-5)
clf.fit(X_train_features, train_df.category_id)

predicted = clf.predict(X_test_features)
result = confusion_matrix(test_df['category_id'], predicted)
print(result)
accuracy = accuracy_score(test_df['category_id'], predicted)
print("accuracy score: " + accuracy.astype(str))
report = classification_report(test_df['category_id'], predicted)
print(report)
print('Total time taken: {0:.2f}s'.format(time.time() - start_time))
