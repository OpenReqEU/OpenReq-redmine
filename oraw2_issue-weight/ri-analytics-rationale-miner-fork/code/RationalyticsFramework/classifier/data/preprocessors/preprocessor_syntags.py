"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of the Rationalytics framework and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

import logging

from base.preprocessor_syntags import do_extract_postags, extract_postags_clause_level, extract_postags_phrase_level, \
    extract_postags_clausephrase_level
from RationalyticsFramework.classifier.data.preprocessors.preprocessor_basic import APreprocessor


class POSTagsExtractor(APreprocessor):
    # CTOR_PARAM_COL_PREFIX = "col_prefix"
    PARAM_EXTENDED_POSTAGS = "extended_postags"

    __col_suffix = "_postags"
    __col_suffix_extended = __col_suffix + "_extended"

    def __init__(self, col_prefix, extended_postags=False):
        self.col_prefix = col_prefix
        self.extended_postags = extended_postags
        self.col_processed = self.get_processed_col_name()

    def get_processed_col_name(self):
        if self.extended_postags:
            col_suffix = POSTagsExtractor.__col_suffix_extended
        else:
            col_suffix = POSTagsExtractor.__col_suffix

        return "%s%s" % (self.col_prefix, col_suffix)

    def get_all_processed_col_names(self):
        return ["%s%s" % (self.col_prefix, POSTagsExtractor.__col_suffix),
                "%s%s" % (self.col_prefix, POSTagsExtractor.__col_suffix_extended)]

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        logging.debug("%s: transform ..." % self.__class__.__name__)

        logging.debug("%s: col_processed: %s" % (self.__class__.__name__, self.col_processed))
        if self.is_data_column_already_processed(df):
            logging.debug("%s: Values already processed." % self.__class__.__name__)
            return df

        txt_list = df[self.col_prefix]
        txt_postags_tokens, txt_postags_strs, txt_postags_strs_extended = do_extract_postags(txt_list)

        #todo we write both postags-column and the ext. variant into the dataframe
        #
        # export both values?
        df.iloc[:self.col_processed] = txt_postags_strs
        df.iloc[:self.col_prefix + POSTagsExtractor.__col_suffix_extended] = txt_postags_strs_extended

        return df


class PENNClauseTagsExtractor(APreprocessor):
    __col_suffix = "_syntags_clause"

    def __init__(self, col_prefix):
        self.col_prefix = col_prefix
        self.col_processed = self.get_processed_col_name()

    def get_processed_col_name(self):
        return "%s%s" % (self.col_prefix, PENNClauseTagsExtractor.__col_suffix)

    def get_all_processed_col_names(self):
        return [self.get_processed_col_name()]

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        logging.debug("%s: transform ..." % self.__class__.__name__)

        logging.debug("%s: col_processed: %s" % (self.__class__.__name__, self.col_processed))
        if self.is_data_column_already_processed(df):
            logging.debug("%s: Values already processed." % self.__class__.__name__)
            return df

        txt_list = df[self.col_prefix]
        txt_tags_strs, x1, x2 = extract_postags_clause_level(txt_list)
        df.iloc[:self.col_processed] = txt_tags_strs[0]

        return df


class PENNPhraseTagsExtractor(APreprocessor):
    __col_suffix = "_syntags_phrase"

    def __init__(self, col_prefix):
        self.col_prefix = col_prefix
        self.col_processed = self.get_processed_col_name()

    def get_processed_col_name(self):
        return "%s%s" % (self.col_prefix, PENNPhraseTagsExtractor.__col_suffix)

    def get_all_processed_col_names(self):
        return [self.get_processed_col_name()]

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        logging.debug("%s: transform ..." % self.__class__.__name__)

        logging.debug("%s: col_processed: %s" % (self.__class__.__name__, self.col_processed))
        if self.is_data_column_already_processed(df):
            logging.debug("%s: Values already processed." % self.__class__.__name__)
            return df

        txt_list = df[self.col_prefix]
        txt_tags_strs, x1, x2 = extract_postags_phrase_level(txt_list)
        df.iloc[:self.col_processed] = txt_tags_strs[0]

        return df


class PENNClausePhraseTagsExtractor(APreprocessor):
    __col_suffix = "_syntags_clause_phrase"

    def __init__(self, col_prefix):
        self.col_prefix = col_prefix
        self.col_processed = self.get_processed_col_name()

    def get_processed_col_name(self):
        return "%s%s" % (self.col_prefix, PENNClausePhraseTagsExtractor.__col_suffix)

    def get_all_processed_col_names(self):
        return [self.get_processed_col_name()]

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        logging.debug("%s: transform ..." % self.__class__.__name__)

        logging.debug("%s: col_processed: %s" % (self.__class__.__name__, self.col_processed))
        if self.is_data_column_already_processed(df):
            logging.debug("%s: Values already processed." % self.__class__.__name__)
            return df

        txt_list = df[self.col_prefix]
        txt_tags_strs, x1, x2 = extract_postags_clausephrase_level(txt_list)
        df.iloc[:self.col_processed] = txt_tags_strs[0]

        return df


class SynTreeHeightExtractor(APreprocessor):
    __col_suffix = "_syn_tree_height"

    def __init__(self, col_prefix):
        self.col_prefix = col_prefix
        self.col_processed = self.get_processed_col_name()

    def get_processed_col_name(self):
        return "%s%s" % (self.col_prefix, SynTreeHeightExtractor.__col_suffix)

    def get_all_processed_col_names(self):
        return [self.get_processed_col_name()]

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        logging.debug("%s: transform ..." % self.__class__.__name__)

        logging.debug("%s: col_processed: %s" % (self.__class__.__name__, self.col_processed))
        if self.is_data_column_already_processed(df):
            logging.debug("%s: Values already processed." % self.__class__.__name__)
            return df

        txt_list = df[self.col_prefix]
        x1, height, x2 = extract_postags_clausephrase_level(txt_list)
        df.iloc[:self.col_processed] = height

        return df


class SynSubTreeCountExtractor(APreprocessor):
    __col_suffix = "_syn_subtree_counts"

    def __init__(self, col_prefix):
        self.col_prefix = col_prefix
        self.col_processed = self.get_processed_col_name()

    def get_processed_col_name(self):
        return "%s%s" % (self.col_prefix, SynSubTreeCountExtractor.__col_suffix)

    def get_all_processed_col_names(self):
        return [self.get_processed_col_name()]

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        logging.debug("%s: transform ..." % self.__class__.__name__)

        logging.debug("%s: col_processed: %s" % (self.__class__.__name__, self.col_processed))
        if self.is_data_column_already_processed(df):
            logging.debug("%s: Values already processed." % self.__class__.__name__)
            return df

        txt_list = df[self.col_prefix]
        x1, x2, subtree_counts = extract_postags_clausephrase_level(txt_list)
        df.iloc[:self.col_processed] = subtree_counts

        return df