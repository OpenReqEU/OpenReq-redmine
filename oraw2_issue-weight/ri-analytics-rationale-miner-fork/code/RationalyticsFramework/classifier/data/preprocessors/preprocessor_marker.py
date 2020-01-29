"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of the Rationalytics framework and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

import logging
from abc import ABC
from inspect import signature

from base.preprocessor_markers import extract_analogy_marker_count, extract_antithesis_marker_count, \
    extract_cause_marker_count, extract_concession_marker_count
from RationalyticsFramework.classifier.data.preprocessors.preprocessor_basic import APreprocessor


class ATextMarker(APreprocessor, ABC):

    def __init__(self, col_prefix, col_suffix,
                 marker_count_strategy):
        self.col_prefix = col_prefix
        self.col_suffix = col_suffix
        self.marker_count_strategy = marker_count_strategy

        self.col_processed = self.get_processed_col_name()

    def _get_processed_col_name(self, col):
        return "%s%s" % (col, self.col_suffix)

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

        marking_out = self.marker_count_strategy(df[self.col_prefix])
        df.insert(len(df.columns), col_processed, marking_out)

        return df

class ATextContextMarker(ATextMarker):

    #todo one possible improvement is: no need for marker_list_getter
    #todo if we select the required data (e.g. contextual data such as title) into the df in the Pre-select step
    def __init__(self, col_prefix, col_suffix,
                 marker_list_getter,
                 marker_count_strategy):

        super().__init__(col_prefix=col_prefix, col_suffix=col_suffix,
                         marker_count_strategy=marker_count_strategy)

        assert(len(signature(marker_count_strategy).parameters) == 2)

        self.marker_list_getter = marker_list_getter
        self.marker_count_strategy = marker_count_strategy

        self.col_processed = self.get_processed_col_name()

    def transform(self, df):
        logging.debug("%s: transform ..." % self.__class__.__name__)

        col_processed = self.get_processed_col_name()
        logging.debug("%s: col_processed: %s" % (self.__class__.__name__, col_processed))
        if self.is_data_column_already_processed(df):
            logging.debug("%s: Values already processed." % self.__class__.__name__)
            return df

        marking_out_list = []
        for row_i,row in df.iterrows():
            marker_list = self.marker_list_getter(row)
            out = self.marker_count_strategy(row[self.col_prefix], marker_list)
            marking_out_list.append(out)

        df.insert(len(df.columns), col_processed, marking_out_list)

        return df
#
# Justification Marker
#

# Relation
# Nb
# Sample indicators
# analogy
# 15
# as a, just as, comes from the same
# antithesis
# 18
# although, even while, on the other hand
# cause
# 14
# because, as a result, which in turn
# concession
# 19
# despite, regardless of, even if
# consequence
# 15
# because, largely because of, as a result of
# contrast
# 8
# but the, on the other hand, but it is the
# evidence
# 7
# attests, this year, according to
# example
# 9
# including, for instance, among the
# explanation-argumentative
# 7
# because, in addition, to comment on the
# purpose
# 30
# trying to, in order to, so as to see
# reason
# 13
# because, because it is, to find a way
# result
# 23
# resulting, because of, as a result o


class AnalogyMarker(ATextMarker):
    __col_suffix = "_jmarker_analogy"

    def __init__(self, col_prefix):
        super().__init__(col_prefix=col_prefix,
                         col_suffix=self.__col_suffix,
                         marker_count_strategy=extract_analogy_marker_count)

class AntiThesisMarker(ATextMarker):
    __col_suffix = "_jmarker_antithesis"

    def __init__(self, col_prefix):
        super().__init__(col_prefix=col_prefix,
                         col_suffix=self.__col_suffix,
                         marker_count_strategy=extract_antithesis_marker_count)


class CauseMarker(ATextMarker):
    __col_suffix = "_jmarker_cause"

    def __init__(self, col_prefix):
        super().__init__(col_prefix=col_prefix,
                         col_suffix=self.__col_suffix,
                         marker_count_strategy=extract_cause_marker_count)

class ConcessionMarker(ATextMarker):
    __col_suffix = "_jmarker_concession"

    def __init__(self, col_prefix):
        super().__init__(col_prefix=col_prefix,
                         col_suffix=self.__col_suffix,
                         marker_count_strategy=extract_concession_marker_count)

class ReasonMarker(ATextMarker):
    __col_suffix = "_jmarker_reason"

    def __init__(self, col_prefix):
        super().__init__(col_prefix=col_prefix,
                         col_suffix=self.__col_suffix,
                         marker_count_strategy=extract_concession_marker_count)

