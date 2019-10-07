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
train_df, test_df, x_train_features, x_test_features = generate_tfidf()

classifier = KNeighborsClassifier(n_neighbors=6)
classifier.fit(x_train_features, train_df.category_id)

predicted = classifier.predict(x_test_features)
# print(predict.shape)
# print(predict)
# print(classifier.predict_proba([[0.9]]))

result = confusion_matrix(test_df['category_id'], predicted)
print(result)
accuracy = accuracy_score(test_df['category_id'], predicted)
print("accuracy score: " + accuracy.astype(str))
report = classification_report(test_df['category_id'], predicted)
print(report)
print('Total time taken: {0:.2f}s'.format(time.time() - start_time))
