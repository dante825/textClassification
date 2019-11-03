from GenTfIdf import generate_tfidf, generate_tf
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import time
import numpy as np

pd.set_option('display.width', 320)
pd.set_option('display.max.columns', 100)
np.set_printoptions(linewidth=320)

start_time = time.time()
# x_train, y_train, x_test, y_test = generate_tfidf()
x_train, y_train, x_test, y_test = generate_tf()
print(x_train.shape)

svd = TruncatedSVD(n_components=4000, n_iter=7, random_state=42, tol=0.0)

x_train_reduced = svd.fit_transform(x_train)
x_test_reduced = svd.transform(x_test)
print(x_train_reduced.shape)

clf = LinearSVC(random_state=0, tol=1e-5)
clf.fit(x_train_reduced, y_train)

predicted = clf.predict(x_test_reduced)
result = confusion_matrix(y_test, predicted)
print(result)
accuracy = accuracy_score(y_test, predicted)
print("accuracy score: " + accuracy.astype(str))
report = classification_report(y_test, predicted)
print(report)
print('Total time taken: {0:.2f}s'.format(time.time() - start_time))