"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of the Rationalytics framework and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

import logging
from abc import ABC, abstractmethod
from collections import deque

from base.preprocessor import SPACY_NLP
from classifier.data import COL_INDEX_SENTENCE, COL_INDEX_COMMENT

class ATextPartsSelector(ABC):
    @abstractmethod
    def get_first_sentences(self, item_list):
        pass

    @abstractmethod
    def get_first_sentence(self, item):
        pass

    @abstractmethod
    def get_last_sentences(self, item_list):
        pass

    @abstractmethod
    def get_last_sentence(self, item):
        pass

    @abstractmethod
    def get_paragraph(self, index_from, index_to):
        pass

class TextPartsSelectorCached(ATextPartsSelector):
    _KEY_CACHE_SENTENCES_FIRST  = "first"
    _KEY_CACHE_SENTENCES_LAST  = "last"

    def __init__(self, df):
        self._df = df

        # local cache
        logging.debug("%s: Create a local cache in ctor.." % self.__class__.__name__)
        self._cache_sentences = {}
        self._cache_sentences[self._KEY_CACHE_SENTENCES_FIRST] = self._get_first_sentences()
        self._cache_sentences[self._KEY_CACHE_SENTENCES_LAST] = self._get_last_sentences()

    def _get_first_sentences(self):
        # if self._KEY_CACHE_SENTENCES_FIRST not in self._cache_sentences:
        # logging.debug("%s: cache first sentences.." % (self.__class__.__name__))
        df = self._df
        df = df[df[COL_INDEX_SENTENCE] == 0]
        # self._cache_sentences[self._KEY_CACHE_SENTENCES_FIRST] = df

        # return self._cache_sentences[self._KEY_CACHE_SENTENCES_FIRST]
        return df

    def _get_last_sentences(self):
        # if self._KEY_CACHE_SENTENCES_LAST not in self._cache_sentences:
        # logging.debug("%s: cache last sentences.." % (self.__class__.__name__))
        df = self._df
        idx = df.groupby([COL_INDEX_COMMENT])[COL_INDEX_SENTENCE].transform(max) == df[COL_INDEX_SENTENCE]
        df = df[idx]
        # self._cache_sentences[self._KEY_CACHE_SENTENCES_LAST] = df

        # return self._cache_sentences[self._KEY_CACHE_SENTENCES_LAST]
        return df

    def get_first_sentences(self, comment_ids):
        df = self._cache_sentences[self._KEY_CACHE_SENTENCES_FIRST]
        return df.loc[df[COL_INDEX_COMMENT].isin(comment_ids)]

    def get_first_sentence(self, comment_id):
        df = self._cache_sentences[self._KEY_CACHE_SENTENCES_FIRST]
        df = df[df[COL_INDEX_COMMENT] == comment_id]
        return df

    def get_last_sentences(self, comment_ids):
        df = self._cache_sentences[self._KEY_CACHE_SENTENCES_LAST]
        return df.loc[df[COL_INDEX_COMMENT].isin(comment_ids)]

    def get_last_sentence(self, comment_id):
        df = self._cache_sentences[self._KEY_CACHE_SENTENCES_LAST]
        df = df[df[COL_INDEX_COMMENT] == comment_id]

        return df

    def get_paragraph(self, index_sentence_from, index_sentence_to):
        raise NotImplementedError()

class TextPartsSelector(ATextPartsSelector):

    def get_first_sentences(self, items):
        res = []
        for item in items:
            sent = self.get_first_sentence(item)
            res.append(sent)

        return res

    def get_first_sentence(self, item):
        text_sentences = SPACY_NLP(item).sents
        return next(text_sentences)

    def get_last_sentences(self, items):
        res = []
        for item in items:
            sent = self.get_last_sentence(item)
            res.append(sent)

        return res

    def get_last_sentence(self, item):
        text_sentences = SPACY_NLP(item).sents
        dd = deque(text_sentences, maxlen=1)
        last_sentence = dd.pop()

        return last_sentence

    def get_first_third_part(self, item):
        text_sentences = SPACY_NLP(item).sents
        n = len(text_sentences)/3
        n = 1 if n == 0 else n

        res = []
        for i in range(1,n):
            s =  next(text_sentences)
            res.append(s)

        return res

    def get_last_third_part(self, item):
        text_sentences = SPACY_NLP(item).sents
        n = len(text_sentences) / 3
        n = 1 if n == 0 else n

        res = []
        for i in range(1, n):
            s = deque(text_sentences, maxlen=1)
            res.append(s)

        return res

    def get_paragraph(self, index_sentence_from, index_sentence_to):
        raise NotImplementedError()