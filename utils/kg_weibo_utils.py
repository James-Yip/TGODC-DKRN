import sys
sys.path.append("..")

import texar as tx
import tensorflow as tf
import numpy as np
import os
import pickle
from preprocess.data_utils import calculate_linsim
from tqdm import tqdm

import gensim


def get_word2vec(w2v_bin):
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_bin, binary=False) #Word2Vec Model
    return word2vec_model


def get_similarity(word1, word2, w2v_model):
    try:
        sim = (w2v_model.n_similarity(word1, word2) + 1)/2
        return sim
    except:
        return 0.5

word2vec_file = "../tx_weibo_data/word2vec_embedding_sim.txt"
w2v_model = get_word2vec(word2vec_file)

def build_keyword_kg(keywords_path,
                     w_mat_output_path, adj_mat_output_path):
    keywords_vocab_list = get_keywords_vocab(keywords_path)
    kg_matrix_size = len(keywords_vocab_list)

    build_adjacency_matrix(keywords_vocab_list, kg_matrix_size, adj_mat_output_path)
    build_weight_matrix(keywords_vocab_list, kg_matrix_size, w_mat_output_path)

def load_keyword_kg(w_mat_path, adj_mat_path):
    kg_adjacency_matrix = load(w_mat_path)
    kg_weight_matrix = load(w_mat_path)

    kg_adjacency_matrix = tf.convert_to_tensor(kg_adjacency_matrix)
    kg_weight_matrix = tf.convert_to_tensor(kg_weight_matrix)

    return kg_weight_matrix, kg_adjacency_matrix


def build_weight_matrix(keywords_vocab_list, kg_matrix_size, output_path):
    print('building weight matrix...')

    kg_weight_matrix = np.zeros(shape=[kg_matrix_size, kg_matrix_size], dtype=np.float32)
    # [1:] -> ignore the <PAD> token when calculating weights (i.e. word similarity)
    for i, kw1 in enumerate(tqdm(keywords_vocab_list[1:])):
        kw1_idx = i + 1
        for j, kw2 in enumerate(keywords_vocab_list[1:(kw1_idx+1)]):
            kw2_idx = j + 1
            kw_sim = get_similarity(kw1, kw2, w2v_model)
            kg_weight_matrix[kw1_idx][kw2_idx] = kw_sim
            kg_weight_matrix[kw2_idx][kw1_idx] = kw_sim

    # print(kg_weight_matrix[:5])
    save(kg_weight_matrix, output_path)

def build_adjacency_matrix(keywords_vocab_list, kg_matrix_size, output_path):
    print('building adjacency matrix...')

    with open("../tx_weibo_data/test/context.txt", "r") as f:
        context_keywords_list = [x.strip().split() for x in f.readlines()]
    with open("../tx_weibo_data/test/keywords.txt", "r") as f:
        next_keywords_list = [x.strip().split() for x in f.readlines()]

    # stoi_dict: dict mapping string(i.e. keyword) into adjacency matrix id
    # vocab_id_to_adj_matrix_id_dict: dict mapping vocab id into adjacency matrix id
    stoi_dict = {}
    vocab_id_to_adj_matrix_id_dict = {}

    kg_adjacency_matrix = np.zeros(shape=[kg_matrix_size, kg_matrix_size], dtype=np.float32)
    # add a large negative value to be easy to generate keyword mask
    kg_adjacency_matrix += -1e8
    # kg_adjacency_matrix += float('-inf')
    for idx, keyword in enumerate(keywords_vocab_list):
        stoi_dict[keyword] = idx

    for context_keywords, next_keywords in \
        tqdm(zip(context_keywords_list, next_keywords_list), total=len(context_keywords_list)):
        for ckw in context_keywords:
            ckw_idx = stoi_dict[ckw]
            for nkw in next_keywords:
                nkw_idx = stoi_dict[nkw]
                kg_adjacency_matrix[ckw_idx][nkw_idx] = 1.

    save(kg_adjacency_matrix, output_path)

def save(content, save_path):
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # save content into certain path
    with open(save_path, 'wb') as f:
        pickle.dump(content, f)

def load(load_path):
    load_dir = os.path.dirname(load_path)
    if not os.path.exists(load_dir):
        os.makedirs(load_dir)
    # save content into certain path
    with open(load_path, 'rb') as f:
        content = pickle.load(f)
    return content

def get_keywords_vocab(keywords_path):
    kw_vocab = tx.data.Vocab(keywords_path)
    keywords_vocab_list = ['<PAD>']
    # [4:] to remove the special tokens('<PAD>', '<BOS>', '<EOS>', '<UNK>') in kw_vocab
    sorted_kw_vocab_items = sorted(kw_vocab.id_to_token_map_py.items(),key=lambda x:x[0])[4:]
    keywords_vocab_list.extend([item[1] for item in sorted_kw_vocab_items])

    return keywords_vocab_list

def get_kg_ids_map(keywords_path, vocab_path):
    vocab_id_to_adj_matrix_id_dict = {}
    keywords_vocab_list = get_keywords_vocab(keywords_path)
    vocab = tx.data.Vocab(vocab_path)

    for idx, keyword in enumerate(keywords_vocab_list):
        vocab_id_to_adj_matrix_id_dict[int(vocab.map_tokens_to_ids_py(keyword))] = idx

    vocab_ids_to_adj_matrix_ids_map = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(
            list(vocab_id_to_adj_matrix_id_dict.keys()), list(vocab_id_to_adj_matrix_id_dict.values()),
            key_dtype=tf.int64, value_dtype=tf.int64
        ),
        default_value = -1
    )

    return vocab_ids_to_adj_matrix_ids_map


if __name__ == '__main__':
    keywords_path = '../tx_weibo_data/train/keywords_vocab.txt'
    w_mat_output_path = '../save/kg/w_mat.pk'
    adj_mat_output_path = '../save/kg/adj_mat.pk'
    build_keyword_kg(keywords_path,
                     w_mat_output_path, adj_mat_output_path)

    # word2vec_file = "../tx_weibo_data/word2vec_embedding_sim.txt"
    # # glove2vec_file = "../tx_weibo_data/glove2vec_embedding_sim.txt"
    #
    # # g2v_mdoel = get_glove2vec(glove2vec_file)
    # w2v_model = get_word2vec(word2vec_file)
    #
    # # print(g2v_mdoel.n_similarity("大佬", "太太"))
    # print((w2v_model.n_similarity("大佬", "太太") + 1)/2)



