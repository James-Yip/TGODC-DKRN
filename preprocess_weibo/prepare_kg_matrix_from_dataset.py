import os
import numpy as np
import json
from tqdm import tqdm

from kg_utils import build_relatedness_matrix, build_adjacent_matrix, save_matrix
if not os.path.exists('../tx_weibo_data'):
    os.mkdir('../tx_weibo_data')
    os.mkdir('../tx_weibo_data/train')
    os.mkdir('../tx_weibo_data/valid')
    os.mkdir('../tx_weibo_data/test')

for stage in ['train', 'valid', 'test']:
    print("Stage: ", stage)
    keywords_vocab_file = open("../tx_weibo_data/{}/keywords_vocab.txt".format(stage), "r")
    keywords_vocab_list = [x.strip() for x in keywords_vocab_file.readlines()]
    keywords_vocab_dict = {}
    inverse_keywords_vocab_dict = {}
    for i in range(len(keywords_vocab_list)):
        keywords_vocab_dict[keywords_vocab_list[i]] = i
        inverse_keywords_vocab_dict[i] = keywords_vocab_list[i]

    context_file = open("../tx_weibo_data/{}/context.txt".format(stage), "r")
    context_list = [x.strip() for x in context_file.readlines()]

    keywords_file = open("../tx_weibo_data/{}/keywords.txt".format(stage), "r")
    keywords_list = [x.strip() for x in keywords_file.readlines()]

    keywords_number = len(keywords_vocab_list)
    kg_adjacent_matrix = np.zeros((keywords_number, keywords_number), dtype=np.int32)
    kg_adjacent_dict = {}
    for i in tqdm(range(len(context_list))):
        context_line = context_list[i]
        keyword_line = keywords_list[i]
        cks = context_line.split()
        nks = keyword_line.split()
        for ck in cks:
            ck_index = keywords_vocab_dict[ck]
            if ck not in kg_adjacent_dict:
                kg_adjacent_dict[ck] = set()
            for nk in nks:
                nk_index = keywords_vocab_dict[nk]
                kg_adjacent_matrix[ck_index][nk_index] = 1
                kg_adjacent_dict[ck].add(nk)
    for key in kg_adjacent_dict.keys():
        kg_adjacent_dict[key] = list(kg_adjacent_dict[key])

    candicate_mask = open("../tx_weibo_data/{}/candicate_mask.txt".format(stage), "w+")
    for context_line in tqdm(context_list):
        ckws = context_line.split()
        context_mask_set = set()
        if len(ckws) == 0:
            for i in range(keywords_number):
                context_mask_set.add(inverse_keywords_vocab_dict[i])

        else:
            for ckw in ckws:
                ckw_index = keywords_vocab_dict[ckw]
                context_mask = kg_adjacent_matrix[ckw_index]
                for i in range(keywords_number):
                    if context_mask[i] == 1:
                        context_mask_set.add(inverse_keywords_vocab_dict[i])

        candicate_mask.write(' '.join(list(context_mask_set)) + '\n')
        context_mask = np.zeros((keywords_number,), dtype=np.int32)
        context_mask_str = []
        for ckw in ckws:
            ckw_index = keywords_vocab_dict[ckw]
            context_mask += kg_adjacent_matrix[ckw_index]

        for i in range(len(context_mask)):
            if context_mask[i] == 0:
                context_mask_str.append("%.8f"%(1e-8))
            else:
                context_mask_str.append(str(1))

        candicate_mask.write(' '.join(context_mask_str) + '\n')


    save_matrix("../tx_weibo_data/{}/kg_adjacent_matrix.txt".format(stage), kg_adjacent_matrix)
    with open("../tx_weibo_data/{}/kg_adjacent_matrix.json".format(stage), "w+") as kg_adjacent_dict_file:
        json.dump(kg_adjacent_dict, kg_adjacent_dict_file, indent=4)