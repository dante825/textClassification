"""
Read the raw dataset downloaded from uci repository into a csv file in 2 columns, namely text and category
"""
import os
import pandas as pd
import logging
import re

logging.basicConfig(level=logging.INFO)


def get_files():
    path = '/home/dante/development/datasets/20_newsgroups/'

    # Getting all the files path
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            files.append(os.path.join(r, file))

    # Read the content of the file
    text = []
    text_class = []
    for f in files:
        # logging.info('reading file: {}'.format(f))
        text_class.append(get_category(f))
        with open(f, 'r', encoding='utf-8', errors='backslashreplace') as reader:
            text_str = reader.read()
            text_str = text_str.encode('utf-8').decode('utf-8', 'replace')
            # text_str = text_str.replace('\n', ' ')
            text.append(pre_process_text(text_str).strip())

    # Create the dataframe from the data read
    list_of_tuple = list(zip(text, text_class))
    df = pd.DataFrame(list_of_tuple, columns=['text', 'category'])
    df.to_csv('../output/20newsGroup.csv', index=False, encoding='utf8')


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
                'X-Newsreader', 'References', 'NNTP-Posting-Host', 'In-reply-to', 'Sender', 'News-Software']
    string_list = string.split('\n')
    new_string = []
    for text in string_list:
        if text.startswith(tuple(prefixes)):
            continue
        else:
            text = re.sub('[^A-Za-z0-9]+', ' ', text)
            new_string.append(text)
    return ' '.join(new_string)


def main():
    get_files()


if __name__ == "__main__":
    main()