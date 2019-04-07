#!/data/pyenv/keras/bin/python

import os
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
import jieba


class Tokenizer:
    data_path = '/data/code/pubu/etc/data'

    def __init__(self):
        # todo: 多个同义词问题
        self.dict_taobao_file = os.path.join(self.data_path, 'taobao-dict.txt')
        self.dict_role_file = os.path.join(self.data_path, 'roles.txt')
        self.synonym_file = os.path.join(self.data_path, 'synonym.txt')
        self.chars_file = os.path.join(self.data_path, 'gbk-chars.txt')
        self.punct_file = os.path.join(self.data_path, 'punct.txt')
        self.synonym = {}
        self.chars = {}
        self.punct = {}
        self.vocabulary = {}
        self._voca_idx = 0
        self.enable_unigram = True
        self._prepare()

    def cut(self, text: str):
        vec = []
        if self.enable_unigram:
            for char in text:
                if char in self.chars or char in self.punct:
                    vec.append(char)
        for word in jieba.cut(text):
            if len(word) > 1:
                vec.append(word)
        return vec

    def _prepare(self):
        jieba.load_userdict(self.dict_taobao_file)
        jieba.load_userdict(self.dict_role_file)
        for w in open(self.chars_file).read().strip():
            self.chars[w] = self._voca_idx
            self._add_voca(w)
        self._load_synonym(self.synonym_file, self.synonym)
        self._load_synonym(self.punct_file, self.punct)
        self._load_dict_voca(self.dict_role_file)
        self._load_dict_voca(self.dict_taobao_file)

    def _load_dict_voca(self, dict_file: str):
        for line in open(dict_file):
            word, _ = line.split(' ', 1)
            self._add_voca(word.strip())

    def _load_synonym(self, path: str, target: dict):
        for line in open(path).readlines():
            wa, wb = map(str.strip, line.strip().split(' ', 1))
            target[wa] = wb
            target[wb] = wa
            # self._add_voca(wa)
            # self._add_voca(wb, incr_idx=False)

    def _add_voca(self, word: str, incr_idx: bool=True):
        idx = self.vocabulary.get(word, None)
        if idx is None:
            idx = self.vocabulary[word] = self._voca_idx
            if incr_idx:
                self._voca_idx += 1
        return idx


tokenizer = Tokenizer()

vectorizer = CountVectorizer(
    input='content', encoding='utf-8',
    strip_accents='unicode',
    analyzer='word', stop_words=None, lowercase=True,
    tokenizer=tokenizer.cut, vocabulary=tokenizer.vocabulary,
    # max_df=0.5, min_df=3, binary=False, dtype=np.int64
)

transformer = TfidfTransformer()

classifier = MLPClassifier(
    solver='lbfgs',  # 'lbfgs' / 'adam' / 'sgd'
    alpha=1e-5,
    hidden_layer_sizes=(5, 2),
    activation='relu',
    batch_size='auto',
)

pipeline = Pipeline([
    ('vectorize', vectorizer),
    ('transform', transformer),
    ('classify', classifier)
])

classifier = MLPClassifier(
    solver='lbfgs',  # 'lbfgs' / 'adam' / 'sgd'
    alpha=1e-5,
    hidden_layer_sizes=(5, 2),
    activation='relu',
    batch_size='auto',
)


def build_samples(train_rate: float=0.7):
    data = []
    kind_list = []
    kind_map = {'goods': 1, 'coupon': 1, 'interact': 2, 'greeting': 3, 'contribute': 2}
    for line in open('/data/code/pubu/doc/comment.classified.csv'):
        _, _, kind, text = line.strip().split(',', 3)
        kind_val = kind_map.get(kind, None)
        if kind_val is None:
            continue
        data.append(text)
        kind_list.append(kind_val)
    total = len(kind_list)
    train_num = int(total * train_rate)
    train_indics = random.sample(range(0, total), train_num)
    train_data, train_target = [], []
    test_data, test_target = [], []
    for i in range(0, total):
        if i in train_indics:
            train_data.append(data[i])
            train_target.append(kind_list[i])
        else:
            test_data.append(data[i])
            test_target.append(kind_list[i])
    train_target = np.array(train_target, dtype=np.uint8)
    test_target = np.array(test_target, dtype=np.uint8)
    return (train_data, train_target), (test_data, test_target)


(train_data, train_target), (test_data, test_target) = build_samples(0.8)


pipeline.fit(train_data, train_target)

predicted = pipeline.predict(test_data)
res = np.mean(predicted == test_target)
print(res)

kind_map = {1: 'goods', 2: 'interact', 3: 'greeting'}
check_samples = random.sample(range(len(test_target)), 50)
for i in check_samples:
    predicted = pipeline.predict([test_data[i]])
    predicted_kind = kind_map[predicted[0]]
    actual_kind = kind_map[test_target[i]]
    print('{}\t{}\t{}'.format(predicted_kind, actual_kind, test_data[i]))
