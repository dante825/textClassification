"""
Dimentional reduction with Sparse PCA
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html#sklearn.decomposition.SparsePCA
"""
from GenTfIdf import generate_tfidf
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import SparsePCA
import pandas as pd
import time
import numpy as np

pd.set_option('display.width', 320)
pd.set_option('display.max.columns', 100)
np.set_printoptions(linewidth=320)

start_time = time.time()
train_df, test_df, x_train_features, x_test_features = generate_tfidf()
print(train_df.shape)
print(x_train_features.shape)

transformer = SparsePCA(n_components=1000, normalize_components=True, random_state=0)

X_train_reduced = transformer.fit_transform(x_train_features.todense())
X_test_reduced = transformer.fit_transform(x_test_features.todense())
print(X_train_reduced.shape)

# classifier = KNeighborsClassifier(n_neighbors=6)
# classifier.fit(X_train_reduced, train_df.category_id)
#
# predicted = classifier.predict(X_test_reduced)
# # print(predict.shape)
# # print(predict)
# # print(classifier.predict_proba([[0.9]]))
#
# result = confusion_matrix(test_df['category_id'], predicted)
# print(result)
# accuracy = accuracy_score(test_df['category_id'], predicted)
# print("accuracy score: " + accuracy.astype(str))
# report = classification_report(test_df['category_id'], predicted)
# print(report)
print('Total time taken: {0:.2f}s'.format(time.time() - start_time))