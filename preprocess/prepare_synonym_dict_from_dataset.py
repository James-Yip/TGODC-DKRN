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

for stage in ['train', 'valid', 'test']:
    print("Stage: ", stage)
    keywords_vocab_file = open("../tx_data/{}/keywords_vocab.txt".format(stage), "r")
    keywords_vocab_list = [x.strip() for x in keywords_vocab_file.readlines()]
    keywords_vocab_dict = {}
    for i in range(len(keywords_vocab_list)):
        keywords_vocab_dict[keywords_vocab_list[i]] = keywords_vocab_list[i]

    synonyms_word_dict = {}
    for key,word in keywords_vocab_dict.items():
        synonym_list = []
        query_word = word.replace(' ', '_')
        for syn in wordnet.synsets(query_word):
            for l in syn.lemmas():
                if l.name() != word:
                    if l.name() in keywords_vocab_dict:
                        synonym_list.append(l.name())
        synonyms_word_dict[word] = synonym_list

    with open("../tx_data/{}/synonyms_word_dict.json".format(stage), "w+") as synonyms_word_dict_file:
         json.dump(synonyms_word_dict, synonyms_word_dict_file, indent=4)