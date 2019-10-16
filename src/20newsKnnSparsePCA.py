"""
Dimentional reduction with Sparse PCA
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html#sklearn.decomposition.SparsePCA
"""
from GenTfIdf import generate_tfidf
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA, IncrementalPCA
import pandas as pd
import time
import numpy as np

pd.set_option('display.width', 320)
pd.set_option('display.max.columns', 100)
np.set_printoptions(linewidth=320)

start_time = time.time()
x_train, y_train, x_test, y_test = generate_tfidf()
print(x_train.shape)

n_components = 100
transformer = IncrementalPCA(n_components=n_components, batch_size=100)

X_train_reduced = transformer.fit_transform(x_train.todense())
X_test_reduced = transformer.fit_transform(x_test.todense())
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
