#!/data/pyenv/keras/bin/python
# coding:utf-8

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

class Segment(object):

    def __init__(self):
        self.stopwords = self._get_stopwards()

    def segment(self, ifile, ofile):
        with open(ifile, 'r', encoding='utf8') as ifp, \
                open(ofile, 'w', encoding='utf8') as ofp:
            while True:
                sentence = ifp.readline()
                if not sentence:
                    break
                segmented = self._segment_sentence(sentence)
                if segmented is not None:
                    ofp.write(segmented)

    def _segment_sentence(self, sentence):
        res = []
        for letter in sentence:
            if letter not in self.stopwords:
                res.append(letter)
        return ' '.join(res) if res else None

    def _get_stopwards(self):
        stopwords = {'【', '】', '/', '，', '-', '——', '(', ')', '[', ']', ' '}
        stopwords.update(set((range(0, 10))))
        return stopwords


def segment():
    _segmentor = Segment()
    _segmentor.segment('/bak/memuu/titles.online.tsv', '/data/code/miner/data/title.segmented.txt')
    print('segment done')


def build_model():
    model = Word2Vec(LineSentence('/data/code/miner/data/title.segmented.txt'), sg=1, window=5, min_count=1, workers=8)
    model.save('/data/code/miner/data/w2v.skip-gram.model')
    model.most_similar()

kh1 = ['蓝', '色', '之', '恋', '持', '久', '保', '湿', '口', '红', ]
kh2= ['可', '爱', '口', '红', '持', '久', '保', '湿', '不', '脱', '色', '学', '生', '款', '滋', '润', '补', '水', '小', '样', '少', '女', '心', '防', '水', '非', '韩', '国', '的']
hj1 = ['葡', '萄', '牙', '进', '口', '红', '酒', '半', '甜', '白', '葡', '萄', '酒', ]
hj2 = ['澳', '大', '利', '亚', '进', '口', '红', '酒', '干', '红', '葡', '萄', '酒', '2', '瓶', ]
kh3 = ['葆', '玛', '蕾', '敦', '持', '久', '保', '湿', '哑', '光', '口', '红', ]

if __name__ == '__main__':
    build_model()