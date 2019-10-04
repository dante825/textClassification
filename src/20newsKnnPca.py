"""
https://scikit-learn.org/stable/modules/multiclass.html
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np

pd.set_option('display.width', 320)
pd.set_option('display.max.columns', 100)
np.set_printoptions(linewidth=320)

df = pd.read_csv('../output/20newsGroup18828.csv')
df['category_id'] = df.category.factorize()[0]
df_2 = df[['text', 'category_id']]

# Some details about the data
# print(df.groupby(['category', 'category_id']).count().sort_values('category_id'))

train_df, test_df = train_test_split(df_2, test_size=0.2)
# print(train_df.groupby(['category', 'category_id']).count().sort_values('category_id'))
# print(test_df.groupby(['category', 'category_id']).count().sort_values('category_id'))
# print(train_df)

tfidf = TfidfVectorizer()
X_train_features = tfidf.fit_transform(train_df.text)
X_test_features = tfidf.transform(test_df.text)
# print(X_train_features.shape)
svd = TruncatedSVD(n_components=1000, n_iter=7, random_state=42, tol=0.0)

svd_train = svd.fit(X_train_features)
svd_test = svd.fit(X_test_features)

classifier = KNeighborsClassifier(n_neighbors=6)
classifier.fit(svd_train, train_df.category_id)

predicted = classifier.predict(svd_test)
# print(predict.shape)
# print(predict)
# print(classifier.predict_proba([[0.9]]))

result = confusion_matrix(test_df['category_id'], predicted)
print(result)
accuracy = accuracy_score(test_df['category_id'], predicted)
print("accuracy score: " + accuracy.astype(str))
report = classification_report(test_df['category_id'], predicted)
print(report)
