import thulac
thu = thulac.thulac()


def split_words(utter):
    return thu.cut(utter, text=True)
# def split_words(dialog):
    # split_dialog = []
    # for utter in dialog:
    #     split_dialog.append(thu.cut(utter, text=True))
    # return split_dialog


keywords_start = 100 # 用于去掉部分高频词汇
keywords_length = 2  # 关键词长度至少为2个汉字
corpus_size = 500000 # 用来筛选的语料规模, 因为速度原因先小一点, 最后可以改为5000000
keywords_min = 500  # 只保留在语料中出现超过2000次的keyword作为候选的keyword
keywords_count = 8   # 只保留keyword个数超过8的dialog

process_num = 12

import pickle
with open('./../tgds_weibo/weibo_corpus_jieba.pk','rb') as f:
    split_corpus = pickle.load(f)

from collections import Counter
keywords = Counter()



# for i in range(len(corpus)):
#     print(i)
#     for dialogue in corpus[i]:
#         for utterance in dialogue:
#             tokens = split_words(dialogue)
#             keywords.update(tokens.split(' '))

for i in range(len(split_corpus)):
    print(i)
    for utterance in split_corpus[i]:
        keywords.update(utterance.split(' '))


keyword_list = []
for k, w in keywords.most_common():
    x = k.split('_')
    if w < keywords_min:
        break
    if len(x) == 2 and x[1] in ['n','v','a'] and len(x[0]) >= keywords_length: # 只保留动词名词形容词
        keyword_list.append(k.split('_')[0])
keyword_list = keyword_list[keywords_start:]

with open('./convai2/candi_keyword.txt', 'w', encoding='UTF-8') as f:
    for keyword in keyword_list:
        f.write(keyword + '\n')


