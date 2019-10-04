import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

raw_data = [
    "the house has a tiny little mouse",
    "the cat saw the mouse",
    "the mouse ran away from the house",
    "the cat finally ate the mouse",
    "the end of the mouse story"
]

# raw_df = pd.read_csv('../output/20newsGroup.csv')
# raw_data = raw_df[['text']]

# Using TfidfTransformer
cv = CountVectorizer()
word_count_vector = cv.fit_transform(raw_data)
print(word_count_vector.shape)

# Compute IDF values
tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(word_count_vector)
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(), columns = ['idf_weights'])
df_idf = df_idf.sort_values(by=['idf_weights'])
# print(df_idf)

# Compute TFIDF score for the documents
# count matrix
count_vector = cv.transform(raw_data)
# tf-idf scores
tf_idf_vector = tfidf_transformer.transform(count_vector)

features_names = cv.get_feature_names()
# get tfidf vector for first document
first_document_vector = tf_idf_vector[0]

# print the scores
df = pd.DataFrame(first_document_vector.T.todense(), index=features_names, columns=['tfidf'])
df = df.sort_values(by=['tfidf'], ascending=False)
print(df)
