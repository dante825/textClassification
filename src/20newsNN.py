"""
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
"""
from GenTfIdf import generate_tfidf, generate_tf, generate_tf_reduced, generate_tfidf_reduced, generate_tfidf_svd, \
    generate_tf_svd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import time
import pandas as pd
import numpy as np

pd.set_option('display.width', 320)
np.set_printoptions(linewidth=320)

start_time = time.time()
# x_train, y_train, x_test, y_test = generate_tfidf()
x_train, y_train, x_test, y_test = generate_tfidf_svd(54)
# x_train, y_train, x_test, y_test = generate_tfidf_reduced(508)
# x_train, y_train, x_test, y_test = generate_tf()
# x_train, y_train, x_test, y_test = generate_tf_svd(54)
# x_train, y_train, x_test, y_test = generate_tf_reduced(1720)
print(x_train.shape)

# Have to manually interrupt it to produce result
clf = MLPClassifier(tol=1e-3)
clf.fit(x_train, y_train)

predicted = clf.predict(x_test)

result = confusion_matrix(y_test, predicted)
print(result)
accuracy = accuracy_score(y_test, predicted)
print("accuracy score: " + accuracy.astype(str))
report = classification_report(y_test, predicted)
print(report)
print('Total time taken: {0:.2f}s'.format(time.time() - start_time))