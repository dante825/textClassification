import pandas as pd
from sklearn import svm

train_df = pd.read_csv('../output/train.csv')
X = train_df[['features']]
y = train_df[['label']]
clf = svm.SVC(gamma='scale')
clf.fit(X, y)

# test_df = pd.read_csv('../output/test.csv')