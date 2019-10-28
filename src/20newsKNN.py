"""
https://scikit-learn.org/stable/modules/multiclass.html
"""
from GenTfIdf import generate_tfidf
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import time

pd.set_option('display.width', 320)
np.set_printoptions(linewidth=320)

start_time = time.time()
x_train, y_train, x_test, y_test = generate_tfidf()
print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

classifier = KNeighborsClassifier(n_neighbors=123)
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
