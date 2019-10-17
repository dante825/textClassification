"""
Read the raw dataset downloaded from uci repository into a csv file in 2 columns, namely text and category
"""
import os
import pandas as pd
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time
import re

nltk.download('stopwords')
logging.basicConfig(level=logging.INFO)


def get_files():
    # path = '/home/dante/development/datasets/20news-18828/'
    path = '/home/db/development/datasets/20news-18828/'

    # Getting all the files path
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            files.append(os.path.join(r, file))

    # Read the content of the file
    text = []
    text_class = []
    for f in files:
        logging.info('reading file: {}'.format(f))
        text_class.append(get_category(f))
        with open(f, 'r', encoding='utf-8', errors='backslashreplace') as reader:
            text_str = reader.read()
            text_str = text_str.encode('utf-8').decode('utf-8', 'replace')
            # text_str = text_str.replace('\n', ' ')
            text.append(pre_process_text(text_str).strip())

    # Create the dataframe from the data read
    list_of_tuple = list(zip(text, text_class))
    df = pd.DataFrame(list_of_tuple, columns=['text', 'category'])
    # Remove rows with empty text cell
    df = df[df.text != '']
    df.to_csv('../output/20newsGroup18828.csv', index=False, encoding='utf8')


def get_category(path):
    """
    Obtain the category of the text by getting it's parent directory name
    :param path: the file path
    :return: the category
    """
    split_filepath = path.split('/')
    return split_filepath[-2]


def pre_process_text(string):
    prefixes = ['Xref', 'Path', 'From', 'Newsgroup', 'Subject', 'Summary', 'Keywords', 'Message-ID', 'Date',
                'Expires', 'Followup-to', 'Distribution', 'Organization','Approved', 'Supercedes', 'Lines',
                'X-Newsreader', 'References', 'NNTP-Posting-Host', 'In-reply-to', 'Sender', 'News-Software',
                'Article-I.D.', 'Article I D']
    string_list = string.split('\n')
    new_line = []
    for line in string_list:
        if line.startswith(tuple(prefixes)):
            continue
        else:
            # Remove email addresses
            tmp_line = re.sub('\S*@\S*\s?', '', line)
            # Remove stopwords
            tmp_line = stopwords_removal(tmp_line)
            # Remove apostrophes s
            tmp_line = re.sub(r"'s", '', tmp_line)
            # Remove all symbols, retain only alphabets and numbers
            tmp_line = re.sub('[^A-Za-z0-9]+', ' ', tmp_line)
            # Lower the case
            tmp_line = tmp_line.lower()
            # Strip the leading and trailing whitespace
            tmp_line = tmp_line.strip()
            # Remove all extra whitespace
            tmp_line = re.sub(r"\s\s+", " ", tmp_line)
            new_line.append(tmp_line)
    new_text = ' '.join(new_line)
    # The maximum number of characters in libre calc cell is 32767
    # new_text = new_text[:32766]
    return new_text


def stopwords_removal(words: str) -> str:
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(words)
    filtered_words = [w for w in tokens if not w in stop_words]
    return ' '.join(filtered_words)


def main():
    start_time = time.time()
    get_files()
    print("Time taken: {0:.2f}s".format(time.time() - start_time))


if __name__ == "__main__":
    main()
