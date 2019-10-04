"""
The base line. A complete program of converting the text to TF-IDF and then apply a
machine learning algorithm on the data.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pandas as pd
import numpy as np

pd.set_option('display.width', 320)
np.set_printoptions(linewidth=320)

df = pd.read_csv('../output/20newsGroup18828.csv')
df['category_id'] = df.category.factorize()[0]

# Some details about the data
# print(df.groupby(['category', 'category_id']).count().sort_values('category_id'))

train_df, test_df = train_test_split(df, test_size=0.2)
print(train_df.groupby(['category', 'category_id']).count().sort_values('category_id'))
print(test_df.groupby(['category', 'category_id']).count().sort_values('category_id'))

# tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, encoding='latin-1', ngram_range=(1,2),
#                         stop_words='english')
tfidf = TfidfVectorizer()
# features = tfidf.fit_transform(df.text).toarray()
X_train_tfidf = tfidf.fit_transform(train_df.text)
print(X_train_tfidf.shape)
X_test_tfidf = tfidf.transform(test_df.text)
# labels = df.category_id
# print(labels)

clf = MultinomialNB().fit(X_train_tfidf, train_df.category_id)
predicted = clf.predict(X_test_tfidf)

result = confusion_matrix(test_df['category_id'], predicted)
print(result)
accuracy = accuracy_score(test_df['category_id'], predicted)
print("accuracy score: " + accuracy.astype(str))
report = classification_report(test_df['category_id'], predicted)
print(report)
