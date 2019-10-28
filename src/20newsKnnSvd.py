"""
https://scikit-learn.org/stable/modules/multiclass.html
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
"""
from GenTfIdf import generate_tfidf
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import time
import numpy as np

pd.set_option('display.width', 320)
pd.set_option('display.max.columns', 100)
np.set_printoptions(linewidth=320)

start_time = time.time()
x_train, y_train, x_test, y_test = generate_tfidf()
print(x_train.shape)

svd = TruncatedSVD(n_components=3, n_iter=7, random_state=42, tol=0.0)

X_train_reduced = svd.fit_transform(x_train)
X_test_reduced = svd.fit_transform(x_test)
print(X_train_reduced.shape)

classifier = KNeighborsClassifier(n_neighbors=123)
classifier.fit(X_train_reduced, y_train)

predicted = classifier.predict(X_test_reduced)

result = confusion_matrix(y_test, predicted)
print(result)
accuracy = accuracy_score(y_test, predicted)
print("accuracy score: " + accuracy.astype(str))
report = classification_report(y_test, predicted)
print(report)
print('Total time taken: {0:.2f}s'.format(time.time() - start_time))
