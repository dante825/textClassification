"""
https://scikit-learn.org/stable/modules/multiclass.html
"""
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.svm import LinearSVC
import time
from GenTfIdf import generate_tfidf
import pandas as pd
import numpy as np

pd.set_option('display.width', 320)
np.set_printoptions(linewidth=320)

start_time = time.time()
train_df, test_df, x_train_features, x_test_features = generate_tfidf()

clf = LinearSVC(random_state=0, tol=1e-5)
clf.fit(x_train_features, train_df.category_id)

predicted = clf.predict(x_test_features)
result = confusion_matrix(test_df.category_id, predicted)
print(result)
accuracy = accuracy_score(test_df.category_id, predicted)
print("accuracy score: " + accuracy.astype(str))
report = classification_report(test_df.category_id, predicted)
print(report)
print('Total time taken: {0:.2f}s'.format(time.time() - start_time))
