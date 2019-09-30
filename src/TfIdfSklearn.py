from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

df = pd.read_csv('../output/20newsGroup18828.csv')
df['category_id'] = df.category.factorize()[0]
train_df, test_df = train_test_split(df, test_size=0.2)

# tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, encoding='latin-1', ngram_range=(1,2),
#                         stop_words='english')
tfidf = TfidfVectorizer()
# features = tfidf.fit_transform(df.text).toarray()
X_train_tfidf = tfidf.fit_transform(train_df.text)
print(X_train_tfidf.shape)
# labels = df.category_id
# print(labels)

clf = MultinomialNB().fit(X_train_tfidf, train_df.category_id)

X_test_tfidf = tfidf.transform(test_df.text)
predicted = clf.predict(X_test_tfidf)
for original, predicted in zip(test_df.category_id, predicted):
    print('%r => %s' % (original, predicted))
# for doc, category in zip(test_df.text, predicted):
#     print('%r => %s' % (doc, test_df.category_id[category]))
