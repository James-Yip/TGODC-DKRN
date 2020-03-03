import random
import os
import shutil

vocab_file = open("../tx_data/vocab.txt", "r")
vocab_list = [x.strip() for x in vocab_file.readlines()]
vocab_set = set()
for vocab in vocab_list:
    vocab_set.add(vocab)
print(len(vocab_set))

new_word_set = set()
for stage in ['train', 'valid', 'test']:
    context_vocab_file = open("../tx_data/{}/context.txt".format(stage), "r")
    context_word_list = []
    for x in context_vocab_file.readlines():
        context_word_list.extend(x.strip().split())


    target_vocab_file = open("../tx_data/{}/keywords.txt".format(stage), "r")
    target_word_list = []
    for x in target_vocab_file.readlines():
        target_word_list.extend(x.strip().split())

    for keyword in context_word_list:
        if keyword not in vocab_set:
            print(keyword)
            new_word_set.add(keyword)

    for keyword in target_word_list:
        if keyword not in vocab_set:
            print(keyword)
            new_word_set.add(keyword)
print(len(vocab_set))
print(len(new_word_set))

vocab_list.extend(list(new_word_set))
print(len(vocab_list))

vocab_file = open("../tx_data/all_vocab.txt", "w")
for keyword in vocab_list:
    vocab_file.write(keyword+'\n')
# vocab_file.writelines(vocab_list)
vocab_file.close()
