# https://medium.com/velotio-perspectives/real-time-text-classification-using-kafka-and-scikit-learn-c2875ad80b3c

from __future__ import division,print_function, absolute_import
from sklearn.datasets import fetch_20newsgroups #built-in dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import pickle

#Defining model and training it
categories = ["talk.politics.misc","misc.forsale","rec.motorcycles",\
"comp.sys.mac.hardware","sci.med","talk.religion.misc"] #http://qwone.com/~jason/20Newsgroups/ for reference
def fetch_train_dataset(categories):
    twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    return twenty_train
def bag_of_words(categories):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(fetch_train_dataset(categories).data)
    pickle.dump(count_vect.vocabulary_, open("vocab.pickle", 'wb'))
    return X_train_counts
def tf_idf(categories):
    tf_transformer = TfidfTransformer()
    return (tf_transformer,tf_transformer.fit_transform(bag_of_words(categories)))
def model(categories):
    clf = MultinomialNB().fit(tf_idf(categories)[1], fetch_train_dataset(categories).target)
    return clf
model = model(categories)
pickle.dump(model,open("model.pickle", 'wb'))
print("Training Finished!")
#Training Finished Here