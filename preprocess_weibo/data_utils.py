import nltk
import os
from nltk.stem import WordNetLemmatizer

import thulac

thu = thulac.thulac()
thu_seq = thulac.thulac(seg_only=True)


# def split_words(dialog):
#     split_dialog = []
#     for utter in dialog:
#         split_dialog.append(thu.cut(utter, text=True))
#     return split_dialog
#
#
# def split_words_seq_only(dialog):
#     split_dialog = []
#     for utter in dialog:
#         split_dialog.append(thu_seq.cut(utter, text=True))
#     return split_dialog

def split_words(utter):
    return thu.cut(utter, text=True)

def split_words_seq_only(utter):
    return thu_seq.cut(utter, text=True)

def kw_tokenize(string):
    return string.split()

#candi_keyword_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'convai2/candi_keyword.txt')
#_candiwords = [x.strip() for x in open(candi_keyword_path, encoding='UTF-8').readlines()]
_candiwords = []

def is_candiword(a):
    if a in _candiwords:
        return True
    return False


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
    return linsim


def is_reach_goal(context, goal):
    context = split_words_seq_only(context)
    if goal in context:
        return True
    # for wd in context:
    #     if is_candiword(wd):
    #         rela = calculate_linsim(wd, goal)
    #         if rela > 0.9:
    #             return True
    return False


def make_context(string):
    string = split_words_seq_only(string)
    context = []
    for word in string:
        if is_candiword(word):
            context.append(word)
    return context


def utter_preprocess(string_list, max_length):
    source, minor_length = [], []
    string_list = string_list[-9:]
    major_length = len(string_list)
    if major_length == 1:
        context = make_context(string_list[-1])
    else:
        context = make_context(string_list[-2] + string_list[-1])
    context_len = len(context)
    while len(context) < 20:
        context.append('<PAD>')
    for string in string_list:
        string = split_words_seq_only(string)
        if isinstance(string,str):
            string = string.split()
        if len(string) > max_length:
            string = string[:max_length]
        string = ['<BOS>'] + string + ['<EOS>']
        minor_length.append(len(string))
        while len(string) < max_length + 2:
            string.append('<PAD>')
        source.append(string)
    while len(source) < 9:
        source.append(['<PAD>'] * (max_length + 2))
        minor_length.append(0)
    return (source, minor_length, major_length, context, context_len)

def pad(sequence, padding_length):
    padded_sequence = sequence.copy()
    while len(padded_sequence) < padding_length:
        padded_sequence.append('<PAD>')
    return padded_sequence