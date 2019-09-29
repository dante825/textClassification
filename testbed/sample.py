# https://medium.com/velotio-perspectives/real-time-text-classification-using-kafka-and-scikit-learn-c2875ad80b3c
# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import pickle

#Defining model and training it
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
#http://qwone.com/~jason/20Newsgroups/ for reference

twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
print(twenty_train.target_names)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
print('X_train_counts shape: ')
print(X_train_counts.shape)

count_vect.vocabulary_.get(u'algorithm')

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
print('X_train_tf shape')
print(X_train_tf.shape)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print('X_train_tfidf shape')
print(X_train_tfidf.shape)

clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))
