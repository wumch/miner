#!/data/pyenv/keras/bin/python

import os
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import parse_stop_words


class Genor(object):

    def __init__(self, lang='chinese'):
        self.tokenizer = Tokenizer(language=lang)
        stemmer = Stemmer(language=lang)
        self.summarizer = Summarizer(stemmer=stemmer)
        self.summarizer.stop_words = self._get_stopwords()

    def gen(self, text: str, sentence_count: int=3) -> str:
        sentence_list = self._gen(text, sentence_count)
        return '\n'.join([str(s) for s in sentence_list])

    def _gen(self, text: str, sentence_count: int) -> list:
        parser = PlaintextParser(text=text, tokenizer=self.tokenizer)
        return self.summarizer(parser.document, sentence_count)

    def _get_stopwords(self):
        root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        path = os.path.join(root_path, 'data', 'stopwords.txt')
        return parse_stop_words(open(path).readlines())


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('usage: %s [sentence-count] <document>' % sys.argv[0])
        sys.exit(0)
    _sentence_count = 5
    if len(sys.argv) == 3:
        _sentence_count = int(sys.argv[1])
    _doc = sys.argv[-1]
    if os.path.isfile(_doc):
        _doc = open(_doc).read()
    _genor = Genor()
    _summary = _genor.gen(_doc, _sentence_count)
    print('-' * 20, 'summary:', '-' * 20)
    print(_summary)
