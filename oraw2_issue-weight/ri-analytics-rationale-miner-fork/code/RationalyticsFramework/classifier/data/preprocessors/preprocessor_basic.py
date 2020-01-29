"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of the Rationalytics framework and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

import logging
from abc import ABC

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from base.preprocessor import remove_punctuation, remove_stopwords, do_lemmatize
from base.preprocessor_basic import extract_word_count


class APreprocessor(BaseEstimator, TransformerMixin, ABC):
    CTOR_PARAM_COL_PREFIX = "col_prefix"

    # todo improve the name of the method
    def are_all_data_columns_already_processed(self, df, col_name_prefix):
        # get all text column names of the preprocessed text
        col_processed = self.get_all_processed_col_names(col_name_prefix)
        df_cols = [col for col in df.columns if col in col_processed]

        return len(df_cols) == len(col_processed)

    def is_data_column_already_processed(self, df):
        return self.get_processed_col_name() in df.columns

    def get_processed_col_name(self):
        pass

    def get_all_processed_col_names(self):
        pass


class TextPreprocessor(APreprocessor):
    """Extract features from each document for DictVectorizer"""
    PARAM_NOPUNCT = "remove_punct"
    PARAM_NOSTOPS = "remove_stops"
    PARAM_DOLEMMATIZE = "do_lemmatize"

    __col_nopunct_suffix = "_nopunct"
    __col_nostops_suffix = "_nostops"
    __col_nostops_nopunct_suffix = "_nostops_nopunct"
    __col_lemmatized_suffix = "_lemmatized"
    __col_lemmatized_nopunct_suffix = "_lemmatized_nopunct"
    __col_lemmatized_nostops_suffix = "_lemmatized_nostops"
    __col_lemmatized_nostops_nopunct_suffix = "_lemmatized_nostops_nopunct"

    #todo file an issue for scikit; if algorithm_params are named without _,
    #todo than the algorithm_params are not propagated through the pipeline
    def __init__(self, col_prefix, remove_punct=False, remove_stops=False, do_lemmatize=False, do_stem=False):

        assert (not (do_lemmatize and do_stem))

        if do_stem:
            raise NotImplementedError()

        if not (remove_stops or remove_punct or do_lemmatize):
            logging.warning("None preprocessing flags set")

        self.col_prefix = col_prefix
        self.remove_punct = remove_punct
        self.remove_stops = remove_stops
        self.do_lemmatize = do_lemmatize

        #
        # prepare all possible columns
        self.col_nopunct = "%s%s" % (self.col_prefix, TextPreprocessor.__col_nopunct_suffix)
        self.col_nostops = "%s%s" % (self.col_prefix, TextPreprocessor.__col_nostops_suffix)
        self.col_nostops_nopunct = "%s%s" % (self.col_prefix, TextPreprocessor.__col_nostops_nopunct_suffix)
        self.col_lemmatized = "%s%s" % (self.col_prefix, TextPreprocessor.__col_lemmatized_suffix)
        self.col_lemmatized_nopunct = "%s%s" % (self.col_prefix, TextPreprocessor.__col_lemmatized_nopunct_suffix)
        self.col_lemmatized_nostops = "%s%s" % (self.col_prefix, TextPreprocessor.__col_lemmatized_nostops_suffix)
        self.col_lemmatized_nostops_nopunct = "%s%s" % (self.col_prefix, TextPreprocessor.__col_lemmatized_nostops_nopunct_suffix)

    def get_processed_col_name(self):
        col_name = self.col_prefix
        if self.remove_punct and not (self.remove_stops and self.do_lemmatize):
            return "%s%s" % (col_name, TextPreprocessor.__col_nopunct_suffix)
        if self.remove_stops and not (self.remove_punct and self.do_lemmatize):
            return "%s%s" % (col_name, TextPreprocessor.__col_nostops_suffix)
        if self.remove_stops and self.remove_punct and not self.do_lemmatize:
            return "%s%s" % (col_name, TextPreprocessor.__col_nostops_nopunct_suffix)
        if self.do_lemmatize and not (self.remove_stops or self.remove_punct):
            return "%s%s" % (col_name, TextPreprocessor.__col_lemmatized_suffix)
        if self.do_lemmatize and self.remove_punct and not self.remove_stops:
            return "%s%s" % (col_name, TextPreprocessor.__col_lemmatized_nopunct_suffix)
        if self.do_lemmatize and self.remove_stops and not self.remove_punct:
            return "%s%s" % (col_name, TextPreprocessor.__col_lemmatized_nostops_suffix)
        if self.do_lemmatize and self.remove_stops and self.remove_punct:
            return "%s%s" % (col_name, TextPreprocessor.__col_lemmatized_nostops_nopunct_suffix)

        return col_name

    def get_all_processed_col_names(self):
        col_name = self.col_prefix
        all_col_names = ["%s%s" % (col_name, TextPreprocessor.__col_nopunct_suffix),
                         "%s%s" % (col_name, TextPreprocessor.__col_nostops_suffix),
                         "%s%s" % (col_name, TextPreprocessor.__col_nostops_nopunct_suffix),
                         "%s%s" % (col_name, TextPreprocessor.__col_lemmatized_suffix),
                         "%s%s" % (col_name, TextPreprocessor.__col_lemmatized_nopunct_suffix),
                         "%s%s" % (col_name, TextPreprocessor.__col_lemmatized_nostops_suffix),
                         "%s%s" % (col_name, TextPreprocessor.__col_lemmatized_nostops_nopunct_suffix)]

        return all_col_names

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        logging.debug("%s: transform ..." % self.__class__.__name__)

        #
        # check whether the column of interest is in df, if true no need to do anything
        col_processed = self.get_processed_col_name()
        logging.debug("%s: col_processed: %s" % (self.__class__.__name__, col_processed))
        if self.is_data_column_already_processed(df):
            logging.debug("%s: Values already processed." % self.__class__.__name__)
            return df

        df = self.transform_data_custom(df)

        return df

    def transform_data_custom(self, df):

        txt_list = df[self.col_prefix]

        txt_nopunct = remove_punctuation(txt_list) if self.remove_punct else None
        txt_nostops = remove_stopwords(txt_list) if self.remove_stops else None
        txt_lemmatized = do_lemmatize(txt_list) if self.do_lemmatize else None

        txt_nostops_nopunct = remove_stopwords(txt_nopunct) if self.remove_stops and self.remove_punct else None

        txt_lemmatized_nopunct = remove_punctuation(txt_lemmatized) if self.do_lemmatize and self.remove_punct else None
        txt_lemmatized_nostops = remove_stopwords(txt_lemmatized) if self.do_lemmatize and self.remove_stops else None
        txt_lemmatized_nostops_nopunct = remove_punctuation(
            txt_lemmatized_nostops) if self.do_lemmatize and self.remove_punct and self.remove_stops else None

        col_names_list = [self.col_nostops, self.col_nopunct, self.col_nostops_nopunct,
                          self.col_lemmatized, self.col_lemmatized_nopunct, self.col_lemmatized_nostops,
                          self.col_lemmatized_nostops_nopunct]
        values_list = [txt_nostops, txt_nopunct, txt_nostops_nopunct,
                       txt_lemmatized, txt_lemmatized_nopunct, txt_lemmatized_nostops,
                       txt_lemmatized_nostops_nopunct]

        df_preprocessed_dict = {col_names_list[i]: values_list[i] for i in range(0, len(col_names_list)) if
                                values_list[i]}

        df_preprocessed = pd.DataFrame(df_preprocessed_dict)
        df_preprocessed.reset_index(inplace=True)
        df.reset_index(inplace=True)

        df2 = pd.concat([df, df_preprocessed], axis=1)

        return df2


class TextLengthExtractor(APreprocessor):
    # CTOR_PARAM_COL_PREFIX = "col_prefix"
    __col_suffix = "_length"

    def __init__(self, col_prefix):
        self.col_prefix = col_prefix
        self.col_processed = self.get_processed_col_name()

    def _get_processed_col_name(self, col):
        return "%s%s" % (col, TextLengthExtractor.__col_suffix)

    def get_processed_col_name(self):
        return self._get_processed_col_name(self.col_prefix)

    def get_all_processed_col_names(self):
        return [self.col_processed]

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        logging.debug("%s: transform ..." % self.__class__.__name__)

        col_processed = self.get_processed_col_name()
        logging.debug("%s: col_processed: %s" % (self.__class__.__name__, col_processed))
        if self.is_data_column_already_processed(df):
            logging.debug("%s: Values already processed." % self.__class__.__name__)
            return df

        logging.debug("%s: col_processed: %s" % (self.__class__.__name__, self.col_processed))
        count_list = extract_word_count(df[self.col_prefix])
        df[self.col_processed] = count_list

        return df


class DenseTransformer(TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self