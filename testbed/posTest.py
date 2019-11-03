import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()


def stopwords_removal(words: str) -> str:
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(words)
    filtered_words = [w for w in tokens if not w in stop_words]
    return ' '.join(filtered_words)


def lemmatization_test(words: str) -> str:
    tokens = word_tokenize(words)
    pos = nltk.pos_tag(tokens)
    print(pos)
    lemmatized = [lemmatizer.lemmatize(word=item[0], pos=get_wordnet_pos(item[1])) for item in pos]
    return ' '.join(lemmatized)


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def main():
    test_data = "a legendary bird is vigorously flying through the sky "
    remove_stop = stopwords_removal(test_data)
    result = lemmatization_test(remove_stop)
    print(result)


if __name__ == "__main__":
    main()