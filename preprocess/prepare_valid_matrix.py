from dataset import dts_Target
from collections import Counter
import pickle
import random
import os
import shutil
from kg_utils import build_relatedness_matrix, build_adjacent_matrix, save_matrix
if not os.path.exists('../tx_data'):
    os.mkdir('../tx_data')
    os.mkdir('../tx_data/train')
    os.mkdir('../tx_data/valid')
    os.mkdir('../tx_data/test')

for stage in ['valid']:
    keywords_vocab_file = open("../tx_data/{}/keywords_vocab.txt".format(stage), "r")
    keywords_list = [x.strip() for x in keywords_vocab_file.readlines()]

    # build adjacent and relatedness for keyword vocab
    adjacent_matrix = build_adjacent_matrix(keywords_list)
    save_matrix("../tx_data/{}/adjacent_matrix.txt".format(stage), adjacent_matrix)

    relatedness_matrix = build_relatedness_matrix(keywords_list)
    save_matrix("../tx_data/{}/relatedness_matrix.txt".format(stage), relatedness_matrix)
