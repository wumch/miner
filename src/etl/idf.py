#!/data/pyenv/keras/bin/python

import sys
import os
import io
import math
import jieba


class Idfer(object):

    def __init__(self, docs, output):
        self.freq = {}
        self.stopwords = {}
        self.docs = docs
        self.out = open(output, 'w') if isinstance(output, str) else output
        self.doc_num = 0

    def calc(self):
        for doc in self._docs():
            self._feed_doc(doc)
            self.doc_num += 1
        if self.doc_num == 0:
            print('no doc', file=sys.stderr)
            sys.exit(1)
        for word, freq in self.freq.items():
            idf_value = math.log(self.doc_num / (freq + 1))
            self._output(word, idf_value)

    def _output(self, word: str, idf_value: float):
        self.out.write('%s %f%s' % (word, idf_value, os.linesep))

    def _docs(self):
        opened = False
        if isinstance(self.docs, io.IOBase):
            fp = self.docs
        elif isinstance(self.docs, str):
            fp = open(self.docs, 'r')
            opened = True
        else:
            return self.docs
        while True:
            doc = fp.readline()
            if not doc:
                break
            yield doc.rstrip()
        if opened:
            fp.close()

    def _feed_doc(self, doc: str):
        for w in set(self._seg(doc)):
            if w in self.stopwords:
                continue
            if w not in self.freq:
                self.freq[w] = 1
            else:
                self.freq[w] += 1

    def _seg(self, doc: str):
        return jieba.lcut(doc)

    def __del__(self):
        self.out.close()


if __name__ == '__main__':
    _root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    _docs = os.path.join(_root_path, 'data', 'comments.txt')
    _outfile = os.path.join(_root_path, 'data', 'comments.idf.txt')
    _idfer = Idfer(docs=_docs, output=_outfile)
    _idfer.calc()
