from dataset import dts_Weibo_Target
from collections import Counter
import pickle
import random
import os
import shutil

keywords_start = 100 # 用于去掉部分高频词汇
keywords_length = 2  # 关键词长度至少为2个汉字
corpus_size = 5000000 # 用来筛选的语料规模, 因为速度原因先小一点, 最后可以改为5000000
keywords_min = 2000  # 只保留在语料中出现超过2000次的keyword作为候选的keyword
keywords_count = 8   # 只保留keyword个数超过8的dialog

if not os.path.exists('../tx_weibo_data'):
    os.mkdir('../tx_weibo_data')
    os.mkdir('../tx_weibo_data/train')
    os.mkdir('../tx_weibo_data/valid')
    os.mkdir('../tx_weibo_data/test')

# need to be trained by ourself
shutil.copy('convai2/source/embedding.txt', '../tx_weibo_data/embedding.txt')

# dataset = dts_Weibo_Target()
# dataset.make_dataset()

split_corpus = pickle.load(open("corpus.pk","rb"))
max_utter = 9
candidate_num = 20
start_corpus_file = open("../tx_weibo_data/start_corpus.txt", "w")
corpus_file = open("../tx_weibo_data/corpus.txt", "w")

from collections import Counter
keywords = Counter()

for i in range(len(split_corpus)):
    for utterance in split_corpus[i]:
        keywords.update(utterance.split(' '))


keyword_list = []
for k, w in keywords.most_common():
    x = k.split('_')
    if w < keywords_min:
        break
    if len(x) == 2 and x[1] in ['n','v','a'] and len(x[0]) >= keywords_length: # 只保留动词名词形容词
        keyword_list.append(k)
keyword_list = keyword_list[keywords_start:]

with open('./convai2/candi_keyword.txt','w',encoding='UTF-8') as f:
    for word in keyword_list:
        f.write(word + '\n')
print('save keyword_list in candi_list.txt')


word_corpus = []
for i, dialog in enumerate(split_corpus):
    tmp_words = []
    for utterance in dialog:
        tmp_words += utterance.split(' ')
    word_corpus.append(tmp_words)


candi_corpus = []
for i, dialog in enumerate(word_corpus):
    if i % 20000 == 0:
        print(i)
    if i > corpus_size:
        break
    knt = 0
    for word in keyword_list:
        if word in dialog:
            knt += 1
    if knt >= keywords_count:
        candi_corpus.append(split_corpus[i])


vocab = set()
last_corpus = []
for i, dialog in enumerate(candi_corpus):
    tmp_dict = {}
    tmp_dict['dialog'] = []
    tmp_dict['kwlist'] = []
    for utter in dialog:
        tmp_list, tmp_utter = [], []
        for word in utter.split(' '):
            if len(word.split('_')) > 1:
                tmp_utter.append(word.split('_')[0])
                vocab.add(word.split('_')[0])
            if word in keyword_list:
                tmp_list.append(word)
        tmp_dict['kwlist'].append(tmp_list)
        tmp_dict['dialog'].append(' '.join(tmp_utter))
    last_corpus.append(tmp_dict)


vocab = list(vocab)
with open('../tx_weibo_data/vocab.txt','w',encoding='UTF-8') as f:
    for word in vocab:
        f.write(word + '\n')
print('save vocab in vocab.txt')


all_data = {'train':[], 'valid':[], 'test':[]}
for id, sess in enumerate(last_corpus):
    type = 'train'
    random_value = random.random()
    if random_value < 0.05:
        type = 'test'
    elif random_value > 0.05 and random_value < 0.10:
        type = 'valid'
    sample = {'dialog':sess['dialog'], 'kwlist':[]}
    for i in range(len(sess['dialog'])):
        # sample['kwlist'].append(keyword_extractor.idf_extract(sess['dialog'][i]))
        new_kwlist = []
        for kw in sess['kwlist'][i]:
            new_kwlist.append(kw.split('_')[0])		
        sample['kwlist'].append(sess['kwlist'][i])
    all_data[type].append(sample)


for stage in ['train', 'valid', 'test']:
    source_file = open("../tx_weibo_data/{}/source.txt".format(stage), "w")
    target_file = open("../tx_weibo_data/{}/target.txt".format(stage), "w")
    context_file = open("../tx_weibo_data/{}/context.txt".format(stage), "w")
    keywords_file = open("../tx_weibo_data/{}/keywords.txt".format(stage), "w")
    label_file = open("../tx_weibo_data/{}/label.txt".format(stage), "w")
    keywords_vocab_file = open("../tx_weibo_data/{}/keywords_vocab.txt".format(stage), "w")
    keywords_list = []
    corpus = []
    keywords_counter = Counter()
    for sample in all_data[stage]:
        corpus += sample['dialog'][1:]
        start_corpus_file.write(sample['dialog'][0]+ '\n')
        for kws in sample['kwlist']:
            keywords_counter.update(kws)
    for kw, _ in keywords_counter.most_common():
        keywords_vocab_file.write(kw + '\n')
        keywords_list.append(kw)
    for sample in all_data[stage]:
        for i in range(2, len(sample['dialog'])):
            if len(sample['kwlist'][i]) > 0:
                source_list = sample['dialog'][max(0, i - max_utter):i]
                source_str = '|||'.join(source_list)
                while True:
                    random_corpus = random.sample(corpus, candidate_num - 1)
                    if sample['dialog'][i] not in random_corpus:
                        break
                corpus_file.write(sample['dialog'][i] + '\n')
                target_list = [sample['dialog'][i]] + random_corpus
                target_str = '|||'.join(target_list)
                source_file.write(source_str + '\n')
                target_file.write(target_str + '\n')
                context_file.write(' '.join(sample['kwlist'][i-2] +
                    sample['kwlist'][i-1]) + '\n')
                keywords_file.write(' '.join(sample['kwlist'][i]) + '\n')
                label_file.write('0\n')
                
    source_file.close()
    target_file.close()
    label_file.close()
    keywords_vocab_file.close()
    context_file.close()


start_corpus_file.close()
corpus_file.close()
