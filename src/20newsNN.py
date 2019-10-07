"""
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
"""
from GenTfIdf import generate_tfidf
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import time
import pandas as pd
import numpy as np

pd.set_option('display.width', 320)
np.set_printoptions(linewidth=320)

start_time = time.time()
train_df, test_df, x_train_features, x_test_features = generate_tfidf()

# Have to manually interrupt it to produce result
clf = MLPClassifier(tol=1e-3)
# clf = MLPClassifier(alpha=1, max_iter=100)
clf.fit(x_train_features, train_df.category_id)

predicted = clf.predict(x_test_features)

result = confusion_matrix(test_df['category_id'], predicted)
print(result)
accuracy = accuracy_score(test_df['category_id'], predicted)
print("accuracy score: " + accuracy.astype(str))
report = classification_report(test_df['category_id'], predicted)
print(report)
print('Total time taken: {0:.2f}s'.format(time.time() - start_time))