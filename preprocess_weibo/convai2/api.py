import os
import pickle
from tqdm import tqdm

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'source')

class dts_Weibo_ConvAI2(object):
    def __init__(self, path=data_path):
        self.path = path
        self.total_dialog = 5000000
        self.dialogs = []
        self.dialogs_pos = []

    def _txt_to_json(self, txt_path, pk_path):
        samples = []
        corpus = []
        with open(txt_path, 'r',encoding='UTF-8') as f:
            tmp_dialog = []
            for line in f.readlines():
                if line == '\n':
                    corpus.append(tmp_dialog.copy())
                    tmp_dialog = []
                else:
                    tmp_dialog.append(line)

        # corpus = corpus[:self.total_dialog]
        with open(pk_path,'rb') as f:
            split_corpus = pickle.load(f)

        for i in range(len(corpus)):
             dialog = corpus[i]
             dialog_pos = split_corpus[i]
             samples.append({
                 'self_persona': [],
                 'other_persona': [],
                 'dialog': dialog,
                 'dialog_pos':dialog_pos,
                 'candidates': []
                 }
             )
        return samples

    def _txt_to_json_from_pk(self, pk_path):
        samples = []
        with open(pk_path,'rb') as f:
            split_corpus = pickle.load(f)
        split_corpus = split_corpus[:self.total_dialog]
        for dialog_pos in tqdm(split_corpus):
            sample = {'self_persona': [],
                      'other_persona': [],
                      'dialog_pos': dialog_pos,
                      'candidates': []
                      }
            dialog = []
            for utterance in dialog_pos:
                words = utterance.split()
                new_utterance = []
                for word in words:
                    new_utterance.append(word.split('_')[0])
                dialog.append(''.join(new_utterance))
            sample['dialog'] = dialog
            samples.append(sample)
        return samples

    def get_data(self):
        # txt_path = os.path.join(self.path, 'weibo_corpus.txt')
        # print("Get dialog from ", txt_path)
        pk_path = os.path.join(self.path, 'weibo_corpus.pk')
        print("Get dialog from ", pk_path)
        # assert os.path.exists(txt_path)
        assert os.path.exists(pk_path)
        # return self._txt_to_json(txt_path, pk_path)
        return self._txt_to_json_from_pk(pk_path)

    def get_dialogs(self):
        if len(self.dialogs) == self.total_dialog:
            return self.dialogs, self.dialogs_pos
        dialogs = []
        dialogs_pos = []
        for sample in self.get_data():
            dialogs.append(sample['dialog'])
            dialogs_pos.append(sample['dialog_pos'])
        self.dialogs = dialogs
        self.dialogs_pos = dialogs_pos

        # dialogs = [sample['dialog'] for sample in self.get_data()]
        return self.dialogs, self.dialogs_pos