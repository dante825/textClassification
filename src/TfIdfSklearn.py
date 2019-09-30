from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

df = pd.read_csv('../output/20newsGroup18828.csv')
df['category_id'] = df.category.factorize()[0]

# tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, encoding='latin-1', ngram_range=(1,2),
#                         stop_words='english')
tfidf = TfidfVectorizer()
features = tfidf.fit_transform(df.text).toarray()
labels = df.category_id
