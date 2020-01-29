"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of the Rationalytics framework and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

import logging

from base.preprocessor_sentiment import do_extract_sentiments
from RationalyticsFramework.classifier.data.preprocessors.preprocessor_basic import APreprocessor


class SentimentExtractor(APreprocessor):
    __col_suffix_pos = "_sentiment_pos"
    __col_suffix_neg = "_sentiment_neg"
    __col_suffix_norm = "_sentiment_norm" # main suffix

    def __init__(self, col_prefix):
        self.col_prefix = col_prefix
        self.col_processed = self.get_processed_col_name()

    def get_processed_col_name(self):
        #todo norm is the main sentiment value; make configurable
        return "%s%s" % (self.col_prefix, SentimentExtractor.__col_suffix_norm)

    def get_all_processed_col_names(self):
        return ["%s%s" % (self.col_prefix, SentimentExtractor.__col_suffix_pos),
                "%s%s" % (self.col_prefix, SentimentExtractor.__col_suffix_neg),
                "%s%s" % (self.col_prefix, SentimentExtractor.__col_suffix_norm)]

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        logging.debug("%s: transform ..." % self.__class__.__name__)

        logging.debug("%s: col_processed: %s" % (self.__class__.__name__, self.col_processed))
        if self.is_data_column_already_processed(df):
            logging.debug("%s: Values already processed." % self.__class__.__name__)
            return df

        txt_list = df[self.col_prefix]
        sentiment_pos, sentiment_neg, sentiment_norm = do_extract_sentiments(txt_list)

        pos_col = "%s%s" % (self.col_prefix, SentimentExtractor.__col_suffix_pos)
        pos_neg = "%s%s" % (self.col_prefix, SentimentExtractor.__col_suffix_neg)

        df[pos_col] = sentiment_pos
        df[pos_neg] = sentiment_neg
        # main col
        df[self.col_processed] = sentiment_norm

        return df