import numpy as np
import collections
import random
import pickle
from convai2 import dts_Weibo_ConvAI2
from extraction import KeywordExtractor
from data_utils import *
from tqdm import tqdm

class dts_Weibo_Target(dts_Weibo_ConvAI2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab = []
        self.sess_set = []

    def get_vocab(self):
        if len(self.vocab) > 0:
            return self.vocab
        counter = collections.Counter()
        dialogs,dialogs_pos = self.get_dialogs()
        for dialog in dialogs_pos:
            for uttr in dialog:
                seq = []
                for word in uttr.split():
                    seq.append(word.split('_')[0])
                counter.update(seq)
                # counter.update(split_words_seq_only(uttr))
        print('total vocab count: ', len(counter.items()))
        vocab = [token for token, times in sorted(list(counter.items()), key=lambda x: (-x[1], x[0]))]
        with open('../tx_data_weibo/vocab.txt','w') as f:
            for word in vocab:
                f.write(word + '\n')
        print('save vocab in vocab.txt')
        self.vocab = vocab
        return vocab

    def get_kwsess(self, vocab):
        if len(self.sess_set) > 0:
            return self.sess_set
        keyword_extractor = KeywordExtractor(vocab)
        corpus = self.get_data()
        sess_set = []
        for sess in corpus:
            data = {}
            data['history'] = ''
            data['history_pos'] = ''
            data['dialog'] = []
            data['dialog_pos'] = []
            for dialog in sess['dialog']:
                data['dialog'].append(dialog)
                data['history'] = data['history'] + ' ' + dialog
            for dialog_pos in sess['dialog_pos']:
                data['dialog_pos'].append(dialog_pos)
                data['history_pos'] = data['history_pos'] + ' ' + dialog_pos
            data['kws'] = keyword_extractor.extract(data['history_pos'])
            # data['kws'] = keyword_extractor.extract(data['history'])
            sess_set.append(data)
        self.sess_set = sess_set
        return sess_set

    def cal_idf(self):
        counter = collections.Counter()
        dialogs, dialogs_pos = self.get_dialogs()
        total = 0.
        for dialog in dialogs_pos:
            for uttr in dialog:
                total += 1
                seq = []
                words = uttr.split()
                for word in words:
                    seq.append(word.split('_')[0])
                counter.update(set(seq))
                # counter.update(set(split_words_seq_only(uttr)))
        idf_dict = {}
        for k,v in counter.items():
            idf_dict[k] = np.log10(total / (v+1.))
        return idf_dict

    def make_dataset(self):
        vocab = self.get_vocab()
        idf_dict = self.cal_idf()
        kw_counter = collections.Counter()
        sess_set = self.get_kwsess(vocab)
        for data in sess_set:
            kw_counter.update(data['kws'])
        kw_freq = {}
        kw_sum = sum(kw_counter.values())
        for k, v in kw_counter.most_common():
            kw_freq[k] = v / kw_sum
        for data in sess_set:
            data['score'] = 0.
            for kw in set(data['kws']):
                data['score'] += kw_freq[kw]
            data['score'] = 0 if len(set(data['kws'])) == 0 else data['score'] / len(set(data['kws']))
        sess_set.sort(key=lambda x: x['score'], reverse=True)

        all_data = {'train':[], 'valid':[], 'test':[]}
        keyword_extractor = KeywordExtractor(idf_dict)
        for id, sess in tqdm(enumerate(sess_set)):
            type = 'train'
            if id < 500:
                type = 'test'
            elif random.random() < 0.05:
                type = 'valid'
            sample = {'dialog':sess['dialog'], 'kwlist':[]}
            for i in range(len(sess['dialog'])):
                # sample['kwlist'].append(keyword_extractor.idf_extract(sess['dialog'][i]))
                sample['kwlist'].append(keyword_extractor.idf_extract(sess['dialog_pos'][i]))
            all_data[type].append(sample)
        pickle.dump(all_data, open('new_weibo_corpus.pk','wb'))
        return all_data