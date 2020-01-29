"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of the Rationalytics framework and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

#INFO in transformers, no private variables should start with an underscore!!!!
# for some reason, the parameters are then not copied

import spacy
import os

from RationalyticsFramework.classifier.data.parts_selector import TextPartsSelector
from sklearn.base import BaseEstimator, TransformerMixin

from RationalyticsFramework.classifier.data.__init__ import COL_INDEX_COMMENT


class DataSelector(BaseEstimator, TransformerMixin):
    CTOR_P_KEY = "key"
    CTOR_P_FEATURE_NAME = "feature_name"

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        return df.loc[:, self.key]

    def get_feature_names(self):
         return [self.key]

class TypedDataSelector(DataSelector):
    CTOR_P_VALUE_TYPE = "value_type"
    CTOR_P_RESHAPE = "reshape"

    def __init__(self, key, value_type, reshape=False):
        super().__init__(key)
        self.reshape = reshape
        self.value_type = value_type

    def transform(self, df):
        if self.reshape:
            return df[self.key].astype(dtype=self.value_type).values.reshape(-1, 1)
        else:
            return df[self.key].astype(dtype=self.value_type)


class DataPreSelector(BaseEstimator, TransformerMixin):
    CTOR_P_KEYS = "keys"
    CTOR_P_TEXT_PARTS_SELECTOR = "text_parts_selector"
    CTOR_P_IS_EVALUATION = "is_evaluation"

    def __init__(self, keys, is_evaluation):
        self.keys = keys
        self.is_evaluation = is_evaluation

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        if self.keys:
            raise NotImplementedError("filtering for keys is currently not impelemted")
            # return data_wrapper.loc[:, self.keys]

        return df

class FirstSentenceSelector(DataPreSelector):
    # SPACY_NLP = spacy.load('en')
    SPACY_NLP = spacy.load('fr')

    def __init__(self, keys, text_parts_selector, is_evaluation=False):
        super().__init__(keys=keys, is_evaluation=is_evaluation)

        self.text_parts_selector = text_parts_selector

    def transform(self, df):
        df = super().transform(df)

        if self.is_evaluation:
            df_first = self.text_parts_selector.get_first_sentences(df[COL_INDEX_COMMENT])
        else:
            #todo fix the parameter
            raise NotImplementedError()
            # df_first = self.text_parts_selector.get_first_sentences(df[self.keys])

        assert (df.shape[0] == df_first.shape[0])

        return df_first #[[self.keys]]

class LastSentenceSelector(DataPreSelector):
    # SPACY_NLP = spacy.load('en')
    SPACY_NLP = spacy.load('fr')

    def __init__(self, keys, text_parts_selector=TextPartsSelector(), is_evaluation=False):
        super().__init__(keys=keys, is_evaluation=is_evaluation)

        self.text_parts_selector = text_parts_selector

    def transform(self, df):
        df = super().transform(df)

        if self.is_evaluation:
            df_last = self.text_parts_selector.get_last_sentences(df[COL_INDEX_COMMENT])
        else:
            raise NotImplementedError()
            #todo fix the parameter
            # df_last = self.text_parts_selector.get_last_sentences(df[self.keys])

        assert (df.shape[0] == df_last.shape[0])

        return df_last #[[self.keys]]

    # used for feature assessment
    # def get_feature_names(self):
    #     return ["ls_" + self.key]

#
# class FirstSentenceSelector(DataPreSelector):
#     SPACY_NLP = spacy.load('en')
#
#     def __init__(self, key, first_sentence_selector_strategy=get_last_sentence):
#         super().__init__(key)
#
#         self._selector=first_sentence_selector_strategy
#
#     def transform(self, data_dict):
#         for row_i, row in data_dict.iterrows():
#             df_first = self._selector(row[COL_INDEX_COMMENT])
#             row[self.key] = df_first[self.key]
#
#         return data_dict[self.key]
#
#     # used for feature assessment
#     def get_feature_names(self):
#         return ["fs_" + self.key]
#
# class LastSentenceSelector(DataPreSelector):
#     SPACY_NLP = spacy.load('en')
#
#     def __init__(self, key, last_sentence_select_strategy=get_first_sentence):
#         super().__init__(key)
#
#         self._selector = last_sentence_select_strategy
#
#     def transform(self, data_dict):
#         for row_i, row in data_dict.iterrows():
#             df_last = DataFrame({self.key : self._selector(row[COL_INDEX_COMMENT])})
#             row[self.key] = df_last[self.key]
#
#         return data_dict[self.key]
#
#     # used for feature assessment
#     def get_feature_names(self):
#         return ["ls_" + self.key]



