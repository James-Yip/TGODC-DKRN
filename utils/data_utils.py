"""Utility functions for data preprocessing
"""

import os
from random import choice
import tokenization
import numpy as np
from collections import Counter
import prettytable as pt
import random


SAVE_KW_TRANS_FILE = 'keywords_transition.txt'
SAVE_UT_TRANS_FILE = 'utterances_transition.txt'
SAVE_KW_TRANS_LABEL_FILE = 'keywords_transition_label.txt'
SAVE_UT_TRANS_LABEL_FILE = 'utterances_transition_label.txt'
SAVE_KW_TRANS_SEG_FILE = 'keywords_transition_segment.txt'
SAVE_CUR_KW_FILE = 'cur_keywords.txt'
SAVE_NEXT_KW_FILE = 'next_keywords.txt'
SAVE_CUR_UT_FILE = 'cur_utterances.txt'
SAVE_NEXT_UT_FILE = 'next_utterances.txt'


def load_file(file_name):
    with open(file_name) as f:
        data = f.readlines()
    return data

def tokenize_sequences(seq_list, tokenizer):
    tokenized_seq_list = []
    for seq in seq_list:
        tmp_list = []
        for token in seq.split():
            tmp_list.extend(tokenizer.tokenize(token))
        tokenized_seq_list.append(['[CLS]'] + tmp_list + ['[SEP]'])
    return tokenized_seq_list


# function to tokenize the keyword context with bert tokenizer
def convert_context_with_bert_tokenizer():
    print('start converting context with bert tokenizer...')
    tokenizer = tokenization.FullTokenizer(
        vocab_file='../tx_data/bertvocab.txt',
        do_lower_case=True)
    for stage in ['train','valid','test']:
        with open(os.path.join('../tx_data', stage, 'context.txt'), 'r') as f:
            original_context_list = f.readlines()
        bert_context_list = tokenize_sequences(original_context_list, tokenizer)
        with open(os.path.join('../tx_data', stage, 'context_bert.txt'), 'w') as f:
            for context in bert_context_list:
                f.write(' '.join(context) + '\n')


def obtain_keyword_transitions_list(keywords_strs, histories_strs):
    keywords = [keywords_str.split() for keywords_str in keywords_strs]
    cur_keywords = keywords[:-1]
    right_next_keywords = keywords[1:]
    wrong_next_keywords = []
    for right_next_keyword in right_next_keywords:
        while True:
            # randomly pick a keyword (from all keywords)
            # that different from the right keyword
            random_select_keyword = choice(keywords)
            if random_select_keyword != right_next_keyword:
                wrong_next_keywords.append(random_select_keyword)
                break

    # histories_lengths: #utterances of each conversation history
    # note: each history contains several utterances separated by '|||'
    histories_lengths = [len(history.split('|||')) for history in histories_strs]
    keywords_transitions = [
        (cur_keyword, right_next_keyword, wrong_next_keyword)
        for idx, (cur_keyword, right_next_keyword, wrong_next_keyword) in enumerate(
            zip(cur_keywords, right_next_keywords, wrong_next_keywords))
        if histories_lengths[idx] <= histories_lengths[idx + 1]
    ]
    return keywords_transitions

def obtain_utterance_transitions_list(histories_strs, target_strs):
    cur_utterances = histories_strs
    right_next_utterances = [target_str.split('|||')[0] for target_str in target_strs]
    wrong_next_utterances = [target_str.strip().split('|||')[np.random.randint(1, 20)] for target_str in target_strs]
    utterances_transitions = [
        (cur_utterance, right_next_utterance, wrong_next_utterance)
        for (cur_utterance, right_next_utterance, wrong_next_utterance) in
            zip(cur_utterances, right_next_utterances, wrong_next_utterances)
    ]
    return utterances_transitions


# import json
# with open("keywords_transitions_dict.json", 'w') as f:
#     json.dump(keywords_transitions_dict, f, indent=4)


# for stage in ['train','valid','test']:
#     with open(os.path.join('../tx_data', stage, SAVE_KW_TRANS_FILE), 'w') as f:
#         for cur_keyword, right_next_keyword, wrong_next_keyword in keywords_transitions_dict[stage]:
#             f.write(' '.join(cur_keyword) + ' ' + ' '.join(right_next_keyword) + '\n')
#             f.write(' '.join(cur_keyword) + ' ' + ' '.join(wrong_next_keyword) + '\n')
#     with open(os.path.join('../tx_data', stage, SAVE_KW_TRANS_LABEL_FILE), 'w') as f:
#         for _ in range(len(keywords_transitions_dict[stage])):
#             f.write('1\n0\n')

# for stage in ['train','valid','test']:
#     with open(os.path.join('../tx_data', stage, SAVE_KW_TRANS_FILE), 'w') as f:
#         with open(os.path.join('../tx_data', stage, SAVE_KW_TRANS_LABEL_FILE), 'w') as labelfile:
#             with open(os.path.join('../tx_data', stage, SAVE_KW_TRANS_SEG_FILE), 'w') as segfile:
#                 for cur_keyword, right_next_keyword, wrong_next_keyword in keywords_transitions_dict[stage]:
#                     for ckw in cur_keyword:
#                         for rnkw in right_next_keyword:
#                             f.write('[CLS] ' + ckw + ' [SEP] ' + rnkw + ' [SEP]' + '\n')
#                             segfile.write('0 0 0 1 1\n')
#                             labelfile.write('1\n')
#                         for wrnw in wrong_next_keyword:
#                             f.write('[CLS] ' +ckw + ' [SEP] ' + wrnw + ' [SEP]' + '\n')
#                             segfile.write('0 0 0 1 1\n')
#                             labelfile.write('0\n')

                # f.write(' '.join(cur_keyword) + ' ' + ' '.join(right_next_keyword) + '\n')
                # f.write(' '.join(cur_keyword) + ' ' + ' '.join(wrong_next_keyword) + '\n')
    # with open(os.path.join('../tx_data', stage, SAVE_KW_TRANS_LABEL_FILE), 'w') as f:
    #     for _ in range(len(keywords_transitions_dict[stage])):
    #         f.write('1\n0\n')

# import tokenization
# tokenizer = tokenization.FullTokenizer(
#     vocab_file='../tx_data/bertvocab.txt',
#     do_lower_case=True)
#
# for stage in ['train','valid','test']:
#     with open(os.path.join('../tx_data', stage, SAVE_KW_TRANS_FILE), 'w') as f:
#         with open(os.path.join('../tx_data', stage, SAVE_KW_TRANS_LABEL_FILE), 'w') as labelfile:
#             with open(os.path.join('../tx_data', stage, SAVE_KW_TRANS_SEG_FILE), 'w') as segfile:
#                 for cur_keyword, right_next_keyword, wrong_next_keyword in keywords_transitions_dict[stage]:
#                     for ckw in cur_keyword:
#                         for rnkw in right_next_keyword:
#                             ckw_tokens = tokenizer.tokenize(ckw)
#                             rnkw_tokens = tokenizer.tokenize(rnkw)
#                             f.write('[CLS] ' + ' '.join(ckw_tokens) + ' [SEP] ' + ' '.join(rnkw_tokens) + ' [SEP]' + '\n')
#                             for _ in range(len(ckw_tokens) + 2):
#                                 segfile.write('0 ')
#                             for _ in range(len(rnkw_tokens)):
#                                 segfile.write('1 ')
#                             segfile.write('1\n')
#                             labelfile.write('1\n')
#                         for wrnw in wrong_next_keyword:
#                             ckw_tokens = tokenizer.tokenize(ckw)
#                             wrnw_tokens = tokenizer.tokenize(wrnw)
#                             f.write('[CLS] ' + ' '.join(ckw_tokens) + ' [SEP] ' + ' '.join(wrnw_tokens) + ' [SEP]' + '\n')
#                             for _ in range(len(ckw_tokens) + 2):
#                                 segfile.write('0 ')
#                             for _ in range(len(wrnw_tokens)):
#                                 segfile.write('1 ')
#                             segfile.write('1\n')
#                             labelfile.write('0\n')


"""functions to generate two datasets to train two types of smoothness classifier
1. **keyword smoothness dataset** for training the keyword transition smoothness classifier
  - It contains keyword transitions which might be smooth or non-smooth.
2. **response smoothness dataset** for training the response smoothness classifier #TODO
- It contains conversation turns which might be also smooth or non-smooth.
"""
# function to generate keyword transition smoothness dataset:
    # generate keyword transition list and label list respectively
    # and save them into 'keywords_transition.txt', 'keywords_transition_label.txt'
def generate_kwts_dataset():
    tokenizer = tokenization.FullTokenizer(
        vocab_file='../tx_data/bertvocab.txt',
        do_lower_case=True)
    # load keyword files of train, valid and test sets
    keywords_dict = {
        stage: load_file(os.path.join('../tx_data', stage, 'keywords.txt'))
        for stage in ['train','valid','test']
    }
    # load conversation history files of train, valid and test sets
    histories_dict = {
        stage: load_file(os.path.join('../tx_data', stage, 'source.txt'))
        for stage in ['train','valid','test']
    }
    keywords_transitions_dict = {
        stage: obtain_keyword_transitions_list(keywords_dict[stage], histories_dict[stage])
        for stage in ['train','valid','test']
    }

    for stage in ['train','valid','test']:
        with open(os.path.join('../tx_data', stage, SAVE_KW_TRANS_FILE), 'w') as f:
            with open(os.path.join('../tx_data', stage, SAVE_KW_TRANS_LABEL_FILE), 'w') as labelfile:
                with open(os.path.join('../tx_data', stage, SAVE_KW_TRANS_SEG_FILE), 'w') as segfile:
                    for cur_keyword, right_next_keyword, wrong_next_keyword in keywords_transitions_dict[stage]:
                        ckw_tokens = []
                        rnkw_tokens = []
                        wrnw_tokens = []
                        for ckw in cur_keyword:
                            ckw_tokens.extend(tokenizer.tokenize(ckw))

                        for rnkw in right_next_keyword:
                            rnkw_tokens.extend(tokenizer.tokenize(rnkw))

                        for wrnw in wrong_next_keyword:
                            wrnw_tokens.extend(tokenizer.tokenize(wrnw))

                        f.write('[CLS] ' + ' '.join(ckw_tokens) + ' [SEP] ' + ' '.join(rnkw_tokens) + ' [SEP]' + '\n')
                        for _ in range(len(ckw_tokens) + 2):
                            segfile.write('0 ')
                        for _ in range(len(rnkw_tokens)):
                            segfile.write('1 ')
                        segfile.write('1\n')
                        labelfile.write('1\n')

                        f.write('[CLS] ' + ' '.join(ckw_tokens) + ' [SEP] ' + ' '.join(wrnw_tokens) + ' [SEP]' + '\n')
                        for _ in range(len(ckw_tokens) + 2):
                            segfile.write('0 ')
                        for _ in range(len(wrnw_tokens)):
                            segfile.write('1 ')
                        segfile.write('1\n')
                        labelfile.write('0\n')
                        # f.write(' '.join(cur_keyword) + ' ' + ' '.join(right_next_keyword) + '\n')
                        # f.write(' '.join(cur_keyword) + ' ' + ' '.join(wrong_next_keyword) + '\n')


# function to generate keyword transition smoothness dataset version_2:
    # generate current keywords list, next keywords list and label list respectively
    # and save them into 'cur_keywords.txt', 'next_keywords.txt', 'keywords_transition_label.txt'
def generate_kwts_dataset_v2():
    # load keyword files of train, valid and test sets
    keywords_dict = {
        stage: load_file(os.path.join('../tx_data', stage, 'keywords.txt'))
        for stage in ['train','valid','test']
    }
    # load conversation history files of train, valid and test sets
    histories_dict = {
        stage: load_file(os.path.join('../tx_data', stage, 'source.txt'))
        for stage in ['train','valid','test']
    }
    keywords_transitions_dict = {
        stage: obtain_keyword_transitions_list(keywords_dict[stage], histories_dict[stage])
        for stage in ['train','valid','test']
    }

    for stage in ['train','valid','test']:
        with open(os.path.join('../tx_data', stage, SAVE_CUR_KW_FILE), 'w') as cur_kw_f:
            with open(os.path.join('../tx_data', stage, SAVE_NEXT_KW_FILE), 'w') as next_kw_f:
                for cur_keyword, right_next_keyword, wrong_next_keyword in keywords_transitions_dict[stage]:
                    cur_kw_f.write(' '.join(cur_keyword) + '\n')
                    next_kw_f.write(' '.join(right_next_keyword) + '\n')
                    cur_kw_f.write(' '.join(cur_keyword) + '\n')
                    next_kw_f.write(' '.join(wrong_next_keyword) + '\n')
        with open(os.path.join('../tx_data', stage, SAVE_KW_TRANS_LABEL_FILE), 'w') as f:
            for _ in range(len(keywords_transitions_dict[stage])):
                f.write('1\n0\n')


# function to generate utterance(response) transition smoothness dataset:
    # generate current utterances list, next utterances list and label list respectively
    # and save them into 'cur_utterances.txt', 'next_utterances.txt', 'utterances_transition_label.txt'
def generate_utts_dataset():
    # load conversation history files of train, valid and test sets
    histories_dict = {
        stage: load_file(os.path.join('../tx_data', stage, 'source.txt'))
        for stage in ['train','valid','test']
    }
    # load utterance files of train, valid and test sets
    targets_dict = {
        stage: load_file(os.path.join('../tx_data', stage, 'target.txt'))
        for stage in ['train','valid','test']
    }
    utterances_transitions_dict = {
        stage: obtain_utterance_transitions_list(histories_dict[stage], targets_dict[stage])
        for stage in ['train','valid','test']
    }

    for stage in ['train','valid','test']:
        with open(os.path.join('../tx_data', stage, SAVE_CUR_UT_FILE), 'w') as cur_utter_f:
            with open(os.path.join('../tx_data', stage, SAVE_NEXT_UT_FILE), 'w') as next_utter_f:
                for cur_utterance, right_next_utterance, wrong_next_utterance in utterances_transitions_dict[stage]:
                    cur_utter_f.write(cur_utterance)
                    next_utter_f.write(right_next_utterance + '\n')
                    cur_utter_f.write(cur_utterance)
                    next_utter_f.write(wrong_next_utterance + '\n')
        with open(os.path.join('../tx_data', stage, SAVE_UT_TRANS_LABEL_FILE), 'w') as f:
            for _ in range(len(utterances_transitions_dict[stage])):
                f.write('1\n0\n')


# function to count the number of keywords in an utterance for all data
def count_keywords_number_in_utterance():
    keywords_len_counter = Counter()
    keywords_list = [
        load_file(os.path.join('../tx_data', stage, 'context.txt'))
        for stage in ['train','valid','test']
    ]
    # for keywords strings in each dataset(train, valid, test)
    for keywords_strs in keywords_list:
        cur_keywords_len = [len(keywords_str.split()) for keywords_str in keywords_strs]
        keywords_len_counter.update(cur_keywords_len)
    table = pt.PrettyTable(['keywords length', 'count'])
    for kw_len, count in keywords_len_counter.most_common():
        table.add_row([kw_len, count])
    print(table)
    averaged_kw_len = sum([k * v for k, v in keywords_len_counter.items()]) / sum(keywords_len_counter.values())
    print('averaged length of keywords: {:.2f}'.format(averaged_kw_len))

# function to generate new context and keyword pairwise dataset for target-guided training:
SAVE_TARGET_GUIDED_CONTEXT_FILE = 'tg_context.txt'
SAVE_TARGET_GUIDED_KEYWORDS_FILE = 'tg_keywords.txt'
SAVE_TARGET_GUIDED_CONTEXT_ST_FILE = 'tg_context_single_turn.txt'
SAVE_TARGET_GUIDED_KEYWORDS_ST_FILE = 'tg_keywords_single_turn.txt'
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
brown_ic = wordnet_ic.ic('ic-brown.dat')

def calculate_linsim(a, b):
    linsim = -1
    syna = wn.synsets(a)
    synb = wn.synsets(b)
    for sa in syna:
        for sb in synb:
            try:
                linsim = max(linsim, sa.lin_similarity(sb, brown_ic))
            except:
                pass
    return max(linsim, 0)


def generate_target_guide_train_dataset():
    # load context files of train, valid and test sets
    context_dict = {
        stage: load_file(os.path.join('../tx_data', stage, 'context.txt'))
        for stage in ['train','valid','test']
    }
    # load keyword files of train, valid and test sets
    keywords_dict = {
        stage: load_file(os.path.join('../tx_data', stage, 'keywords.txt'))
        for stage in ['train','valid','test']
    }
    # load conversation history files of train, valid and test sets
    histories_dict = {
        stage: load_file(os.path.join('../tx_data', stage, 'source.txt'))
        for stage in ['train','valid','test']
    }

    dialogue_context_keywords_dict = {
        stage: obtain_dialogue_context_keyword_list(context_dict[stage], keywords_dict[stage], histories_dict[stage])
        for stage in ['train','valid','test']
    }

    for stage in ['train','valid','test']:
        with open(os.path.join('../tx_data', stage, SAVE_TARGET_GUIDED_CONTEXT_FILE), 'w') as tg_context_f:
            with open(os.path.join('../tx_data', stage, SAVE_TARGET_GUIDED_KEYWORDS_FILE), 'w') as tg_keyword_f:
                for dialogue_context_keywords_list in dialogue_context_keywords_dict[stage]:
                    last_context_keywords = dialogue_context_keywords_list[-1]
                    last_context = last_context_keywords[0]
                    last_keywords = last_context_keywords[1]

                    temp_sim = 0
                    target_keyword = last_keywords[0]
                    for keyword in last_keywords:
                        total_sim = 0
                        for context_word in last_context:
                            total_sim += calculate_linsim(keyword, context_word)
                        if total_sim > temp_sim:
                            target_keyword = keyword

                    context_list = []
                    keyword_list = []
                    for dialogue_context_keywords in dialogue_context_keywords_list:
                        dialogue_context_keywords[0].insert(0,target_keyword)
                        context_list.append(' '.join(dialogue_context_keywords[0]))
                        keyword_list.append(' '.join(dialogue_context_keywords[1]))
                    context_str = ' ||| '.join(context_list)
                    keyword_str = ' ||| '.join(keyword_list)
                    tg_context_f.write(context_str + '\n')
                    tg_keyword_f.write(keyword_str + '\n')


def generate_target_guide_single_turn_train_dataset():
    # load context files of train, valid and test sets
    context_dict = {
        stage: load_file(os.path.join('../tx_data', stage, 'context.txt'))
        for stage in ['train','valid','test']
    }
    # load keyword files of train, valid and test sets
    keywords_dict = {
        stage: load_file(os.path.join('../tx_data', stage, 'keywords.txt'))
        for stage in ['train','valid','test']
    }
    # load conversation history files of train, valid and test sets
    histories_dict = {
        stage: load_file(os.path.join('../tx_data', stage, 'source.txt'))
        for stage in ['train','valid','test']
    }

    dialogue_context_keywords_dict = {
        stage: obtain_dialogue_context_keyword_list(context_dict[stage], keywords_dict[stage], histories_dict[stage])
        for stage in ['train','valid','test']
    }

    for stage in ['train','valid','test']:
        with open(os.path.join('../tx_data', stage, SAVE_TARGET_GUIDED_CONTEXT_ST_FILE), 'w') as tg_context_f:
            with open(os.path.join('../tx_data', stage, SAVE_TARGET_GUIDED_KEYWORDS_ST_FILE), 'w') as tg_keyword_f:
                for dialogue_context_keywords_list in dialogue_context_keywords_dict[stage]:
                    last_context_keywords = dialogue_context_keywords_list[-1]
                    last_context = last_context_keywords[0]
                    last_keywords = last_context_keywords[1]

                    temp_sim = 0
                    target_keyword = last_keywords[0]
                    for keyword in last_keywords:
                        total_sim = 0
                        for context_word in last_context:
                            total_sim += calculate_linsim(keyword, context_word)
                        if total_sim > temp_sim:
                            target_keyword = keyword

                    for dialogue_context_keywords in dialogue_context_keywords_list:
                        dialogue_context_keywords[0].insert(0,target_keyword)
                        context_str = ' '.join(dialogue_context_keywords[0])
                        keyword_str = ' '.join(dialogue_context_keywords[1])
                        tg_context_f.write(context_str + '\n')
                        tg_keyword_f.write(keyword_str + '\n')


def obtain_dialogue_context_keyword_list(context_strs, keywords_strs, histories_strs):
    dialogue_context_keyword_list = []
    contexts = [context_str.split() for context_str in context_strs]
    keywords = [keywords_str.split() for keywords_str in keywords_strs]
    # histories_lengths: #utterances of each conversation history
    # note: each history contains several utterances separated by '|||'
    histories_lengths = [len(history.split('|||')) for history in histories_strs]
    single_dialogue_context_keywords_list = []
    for idx in range(len(histories_lengths) - 1):
        if histories_lengths[idx] <= histories_lengths[idx+1]:
            single_dialogue_context_keywords_list.append([contexts[idx], keywords[idx]])
        else:
            single_dialogue_context_keywords_list.append([contexts[idx], keywords[idx]])
            dialogue_context_keyword_list.append(single_dialogue_context_keywords_list)
            single_dialogue_context_keywords_list = []
    single_dialogue_context_keywords_list.append([contexts[-1], keywords[-1]])
    dialogue_context_keyword_list.append(single_dialogue_context_keywords_list)

    return dialogue_context_keyword_list


def generate_target_keywords_for_simulation():
    keywords_file = '../tx_data/test/keywords_vocab.txt'
    target_file = '../tx_data/target_keywords_for_simulation.txt'
    with open(keywords_file, 'r') as f:
        target_set = [x.strip() for x in f.readlines()]
    target_keywords = random.sample(target_set, 500)
    with open(target_file, 'w') as f:
        for kw in target_keywords:
            f.write(kw + '\n')


def generate_start_utterances_for_simulation():
    start_corpus_file = '../tx_data/start_corpus.txt'
    target_corpus_file = '../tx_data/sample_start_corpus.txt'
    with open(start_corpus_file, 'r') as f:
        start_set = [x.strip() for x in f.readlines()]
    start_utterances = random.sample(start_set, 5)
    with open(target_corpus_file, 'w') as f:
        for utterance in start_utterances:
            f.write(utterance + '\n')



if __name__ == '__main__':
    # generate_kwts_dataset()
    # convert_context_with_bert_tokenizer()
    # generate_kwts_dataset_v2()
    # generate_utts_dataset()
    # count_keywords_number_in_utterance()
    # generate_target_guide_train_dataset()
    # generate_target_guide_single_turn_train_dataset()
    # generate_target_keywords_for_simulation()
    generate_start_utterances_for_simulation()
