from data_utils import *

keywords_length = 2 # 关键词长度至少为2个汉字

class KeywordExtractor():
    def __init__(self, idf_dict = None):
        self.idf_dict = idf_dict

    @staticmethod
    def is_keyword_tag(word, tag):
        # x = keywords.split('_')
        # return len(x) == 2 and (x[1] == 'n' or x[1] == 'v' or x[1] == 'a') and len(x[0]) >= keywords_length
        return (tag == 'n' or tag == 'v' or tag == 'a') and len(word) >= keywords_length

    @staticmethod
    def cal_tag_score(tag):
        if tag.startswith('v'):
            return 1.
        if tag.startswith('n'):
            return 2.
        if tag.startswith('a'):
            return 0.5
        return 0.

    def idf_extract(self, string, con_kw = None):
        # temp_tokens = split_words(string)
        temp_tokens = string.split()
        seq_len = len(temp_tokens)
        tokens = []
        for token in temp_tokens:
            x = token.split('_')
            tokens.append((x[0],x[1]))

        # source = split_words_seq_only(string)
        source = []
        for token in temp_tokens:
            x = token.split('_')
            source.append(x[0])
        candi = []
        result = []
        for i, (word, tag) in enumerate(tokens):
            score = self.cal_tag_score(tag)
            if not is_candiword(source[i]) or score == 0.:
                continue
            if con_kw is not None and source[i] in con_kw:
                continue
            score *= source.count(source[i])
            score *= 1 / seq_len
            score *= self.idf_dict[source[i]]
            candi.append((source[i], score))
            if score > 0.15:
                result.append(source[i])
        return list(set(result))


    def extract(self, string):
        # temp_tokens = split_words(string)
        temp_tokens = string.split()
        tokens = []
        for token in temp_tokens:
            x = token.split('_')
            tokens.append((x[0],x[1]))

        # source = split_words_seq_only(string)
        source = []
        for token in temp_tokens:
            x = token.split('_')
            source.append(x[0])
        kwpos_alters = []
        for i, (word, tag) in enumerate(tokens):
            if source[i] and self.is_keyword_tag(word,tag):
                kwpos_alters.append(i)
        kwpos, keywords = [], []
        for id in kwpos_alters:
            if is_candiword(source[id]):
                keywords.append(source[id])
        return list(set(keywords))