import json
import os
import time
import requests
import numpy as np
from queue import Queue
from tqdm import tqdm


def get_relatedness_weight(word1, word2):
    key_word1 = word1.replace(' ', '_')
    key_word2 = word2.replace(' ', '_')
    while True:
        try:
            relatedness_url = "http://172.18.242.134:8084/relatedness?node1=/c/en/%s&node2=/c/en/%s" % (key_word1, key_word2)
            relatedness_json = requests.get(relatedness_url).json()
            break
        except Exception as e:
            print(e)

    relatedness = relatedness_json['value']
    return relatedness


def is_adjacent(word1, word2):
    key_word1 = word1.replace(' ', '_')
    key_word2 = word2.replace(' ', '_')
    while True:
        try:
            query_url = "http://172.18.242.134:8084/query?node=/c/en/%s&other=/c/en/%s" % (key_word1, key_word2)
            query_json = requests.get(query_url).json()
            break
        except Exception as e:
            print(e)

    related_edges = query_json['edges']
    for edge in related_edges:
        end_item = edge['end']['term'].split("/")[3]
        start_item = edge['start']['term'].split("/")[3]
        if start_item == key_word1 and end_item == key_word2:
            return True

        if start_item == key_word2 and end_item == key_word1:
            return True

    return False


def get_edges(word):
    key_word = word.replace(' ', '_')
    while True:
        try:
            query_url = "http://172.18.242.134:8084/c/en/%s" % (key_word)
            query_json = requests.get(query_url).json()
            break
        except Exception as e:
            print(e)

    related_edges = query_json['edges']
    edges = []
    for edge in related_edges:
        end_item_splits = edge['end']['term'].split("/")
        start_item_splits = edge['start']['term'].split("/")
        if end_item_splits[2] != 'en' or start_item_splits[2] != 'en':
            continue
        end_item = end_item_splits[3]
        start_item = start_item_splits[3]
        if start_item == key_word:
            edges.append(end_item.replace('_', ' '))

        if end_item == key_word:
            edges.append(start_item.replace('_', ' '))

    return edges

# def get_relatedness_weight(word1, word2):
#     key_word1 = word1.replace(' ', '_')
#     key_word2 = word2.replace(' ', '_')
#     try:
#         relatedness_url = "http://api.conceptnet.io/relatedness?node1=/c/en/%s&node2=/c/en/%s" % (key_word1, key_word2)
#         relatedness_json = requests.get(relatedness_url).json()
#     except:
#         time.sleep(60)
#         relatedness_url = "http://api.conceptnet.io/relatedness?node1=/c/en/%s&node2=/c/en/%s" % (key_word1, key_word2)
#         relatedness_json = requests.get(relatedness_url).json()
#
#     relatedness = relatedness_json['value']
#     return relatedness
#
#
# def is_adjacent(word1, word2):
#     key_word1 = word1.replace(' ', '_')
#     key_word2 = word2.replace(' ', '_')
#     try:
#         query_url = "http://api.conceptnet.io/query?node=/c/en/%s&other=/c/en/%s" % (key_word1, key_word2)
#         query_json = requests.get(query_url).json()
#     except:
#         time.sleep(60)
#         query_url = "http://api.conceptnet.io/query?node=/c/en/%s&other=/c/en/%s" % (key_word1, key_word2)
#         query_json = requests.get(query_url).json()
#
#     related_edges = query_json['edges']
#     for edge in related_edges:
#         end_item = edge['end']['term'].split("/")[3]
#         start_item = edge['start']['term'].split("/")[3]
#         if start_item == key_word1 and end_item == key_word2:
#             return True
#
#         if start_item == key_word2 and end_item == key_word1:
#             return True
#
#     return False
#
#
# def get_edges(word):
#     key_word = word.replace(' ', '_')
#     try:
#         query_url = "http://api.conceptnet.io/c/en/%s" % (key_word)
#         query_json = requests.get(query_url).json()
#     except:
#         time.sleep(60)
#         query_url = "http://api.conceptnet.io/c/en/%s" % (key_word)
#         query_json = requests.get(query_url).json()
#
#     related_edges = query_json['edges']
#     edges = []
#     for edge in related_edges:
#         end_item_splits = edge['end']['term'].split("/")
#         start_item_splits = edge['start']['term'].split("/")
#         if end_item_splits[2] != 'en' or start_item_splits[2] != 'en':
#             continue
#         end_item = end_item_splits[3]
#         start_item = start_item_splits[3]
#         if start_item == key_word:
#             edges.append(end_item.replace('_', ' '))
#
#         if end_item == key_word:
#             edges.append(start_item.replace('_', ' '))
#
#     return edges


def build_adjacent_matrix(word_list):
    word_list_len = len(word_list)
    adjacent_matrix = np.zeros((word_list_len, word_list_len), dtype=np.int32)
    for i in tqdm(range(word_list_len)):
        for j in tqdm(range(word_list_len)):
            if is_adjacent(word_list[i], word_list[j]):
                adjacent_matrix[i][j] = int(1)
    return adjacent_matrix


def build_relatedness_matrix(word_list):
    word_list_len = len(word_list)
    relatedness_matrix = np.zeros((word_list_len, word_list_len))
    for i in tqdm(range(word_list_len)):
        for j in tqdm(range(word_list_len)):
            relatedness = get_relatedness_weight(word_list[i],word_list[j])
            relatedness_matrix[i][j] = relatedness
    return relatedness_matrix


def save_matrix(fname, matrix):
    np.savetxt(fname, matrix, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='',
               comments='# ', encoding='utf-8')


def load_matrix(fname, dtype=np.float):
    matrix = np.loadtxt(fname, dtype=dtype, comments='#', delimiter=None, converters=None,
                        skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='utf-8')
    return matrix


def breath_first_search(adjacent_matrix, start_word_id, target_word_id):
    word_list_len = len(adjacent_matrix)
    visited = np.zeros((word_list_len,), dtype=np.int32)
    previous = np.zeros((word_list_len,), dtype=np.int32)

    if adjacent_matrix is None or start_word_id < 0 or \
            start_word_id >= word_list_len or target_word_id < 0 or\
            target_word_id >= word_list_len:
        return None

    if start_word_id == target_word_id:
        return [start_word_id]

    queue = Queue()
    queue.put(start_word_id)
    visited[start_word_id] = 1
    previous[start_word_id] = start_word_id
    find_target = False
    while not queue.empty():
        cur = queue.get()
        if cur == target_word_id:
            find_target = True
            break
        adjacent_list = adjacent_matrix[cur]
        for id, adjacent in enumerate(adjacent_list):
            if visited[id] == 0 and adjacent == 1:
                visited[id] = 1
                queue.put(id)
                previous[id] = cur

    word_id_list = []
    if find_target == False:
        for id in range(word_list_len):
            word_id_list.append(id)
    else:
        cur = target_word_id
        word_id_list.append(cur)
        while cur != start_word_id:
            cur = previous[cur]
            word_id_list.append(cur)

    return word_id_list


def get_relatedness_mask(adjacent_matrix,relatedness_matrix,  start_word_id, target_word_id):
    related_word_id_list = breath_first_search(adjacent_matrix, start_word_id, target_word_id)
    relatedness_mask = np.zeros((len(relatedness_matrix)))
    for word_id in related_word_id_list:
        relatedness_mask[word_id] = relatedness_matrix[target_word_id][word_id]

    return relatedness_mask



# def get_relatedness_mask(target_word_id, related_word_id_list, relatedness_matrix):
#     relatedness_mask = np.zeros((len(relatedness_matrix)))
#     for word_id in related_word_id_list:
#         relatedness_mask[word_id] = relatedness_matrix[target_word_id][word_id]
#
#     return relatedness_mask


if __name__ == "__main__":
    word_list = ["example", "apple", "fruit"]
    am = build_adjacent_matrix(word_list)
    save_matrix("test.txt", am)
    am = load_matrix("test.txt",dtype=np.int32)
    print(am)
    print(len(am[0]))
    visited = np.zeros((10,))
    print(visited.shape)
    print(breath_first_search(am,1,2))
    print(am[0][0])