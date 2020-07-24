from extractor.utils import get_extract_label
from os import listdir
from os.path import isfile
from os.path import join

import json
import pickle


def load_data(file_path):
    """
    :param file_path:
    :return:
    """
    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]

    # Extract articles and abstracts
    articles = list()
    abstracts = list()

    for file_name in file_names:
        f = open(file_path + file_name)
        a = json.loads(f.read())
        articles.append(a['article'])
        abstracts.append(a['abstract'])

    return articles, abstracts


def create_training_dictionaries(output_data_path="data/training_data/", input_file_path="data/dev_raw/"):
    """
    :param output_data_path:
    :param input_file_path:
    :return:
    """
    documents, summaries = load_data(file_path=input_file_path)

    training_dictionaries = [
        {'document': doc,
         'summary': summ,
         'extraction_label': get_extract_label(doc, summ)
         } for doc, summ in zip(documents, summaries)
    ]

    pickle.dump(training_dictionaries, open(output_data_path + "training_dictionaries.pkl", "wb"))


def load_training_dictionaries(input_file="data/training_data/training_dictionaries.pkl"):
    """
    :param input_file:
    :return:
    """
    training_dictionaries = pickle.load(open(input_file, "rb"))
    return training_dictionaries
