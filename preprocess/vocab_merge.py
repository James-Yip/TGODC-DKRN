import os
import numpy as np
import json
from tqdm import tqdm

import nltk
from nltk.corpus import wordnet


if not os.path.exists('../tx_data'):
    os.mkdir('../tx_data')
    os.mkdir('../tx_data/train')
    os.mkdir('../tx_data/valid')
    os.mkdir('../tx_data/test')

vocab_file = open("../tx_data/vocab.txt", "r")
vocab_list = [x.strip() for x in vocab_file.readlines()]
vocab_set = set()
for vocab in vocab_list:
    vocab_set.add(vocab)

for stage in ['train', 'valid', 'test']:
    print("Stage: ", stage)
    keywords_vocab_file = open("../tx_data/{}/keywords_vocab.txt".format(stage), "r")
    keywords_vocab_list = [x.strip() for x in keywords_vocab_file.readlines()]
    for keywords_vocab in keywords_vocab_list:
        vocab_set.add(keywords_vocab)


new_vocab_file = open("../tx_data/new_vocab.txt", "w+")
new_vocab_list = list(vocab_set)
for vocab in new_vocab_list:
    new_vocab_file.write(vocab + "\n")
new_vocab_file.close()