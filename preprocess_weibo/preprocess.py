import pickle
import thulac
import multiprocessing as mp

total_dialog = 5000000
# start_keyword = 100
# total_keyword = 1000
process_num = 12
thu = thulac.thulac()


def split_words(dialog):
    split_dialog = []
    for utter in dialog:
        split_dialog.append(thu.cut(utter, text=True))
    return split_dialog


if __name__ == '__main__':
    corpus = []

    with open('./convai2/source/weibo_corpus.txt', encoding='UTF-8') as f:
        tmp_dialog = []
        for line in f.readlines():
            if line == '\n':
                corpus.append(tmp_dialog.copy())
                tmp_dialog = []
            else:
                tmp_dialog.append(line)

    pool = mp.Pool(processes=process_num)
    split_corpus = pool.map(split_words, corpus[:total_dialog])

    with open('weibo_corpus.pk', 'wb') as f:
        pickle.dump(corpus, f)
