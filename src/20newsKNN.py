"""
https://scikit-learn.org/stable/modules/multiclass.html
"""
from GenTfIdf import generate_tfidf, generate_tf, generate_tf_reduced, generate_tfidf_reduced, generate_tfidf_svd, \
    generate_tf_svd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import time
import math

pd.set_option('display.width', 320)
np.set_printoptions(linewidth=320)

start_time = time.time()
# x_train, y_train, x_test, y_test = generate_tfidf()
# x_train, y_train, x_test, y_test = generate_tfidf_svd(4000)
# x_train, y_train, x_test, y_test = generate_tfidf_reduced(50)
# x_train, y_train, x_test, y_test = generate_tf()
x_train, y_train, x_test, y_test = generate_tf_svd(4000)
# x_train, y_train, x_test, y_test = generate_tf_reduced(1000)
print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

k = int(math.sqrt(x_train.shape[1]))
# Rule of thumb for k is sqrt(sample size)
classifier = KNeighborsClassifier(n_neighbors=k)
classifier.fit(x_train, y_train)

predicted = classifier.predict(x_test)
# print(predict.shape)
# print(predict)
# print(classifier.predict_proba([[0.9]]))

result = confusion_matrix(y_test, predicted)
print(result)
accuracy = accuracy_score(y_test, predicted)
print("accuracy score: " + accuracy.astype(str))
report = classification_report(y_test, predicted)
print(report)
print('Total time taken: {0:.2f}s'.format(time.time() - start_time))
