"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of the Rationalytics framework and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

import logging

import pandas as pd
from RationalyticsFramework.classifier.data.parts_selector import TextPartsSelectorCached
from RationalyticsFramework.classifier.data.sampler import ASampler

from RationalyticsFramework.classifier.data.data_wrapper import DataWrapper, UserComments
# user rationale truthsets
from RationalyticsFramework.classifier.data.__init__ import CSV_SEPARATOR, DataSampleConfig, GRANULARITY_COMMENT, GRANULARITY_SENTENCE, \
    GRANULARITY_LEVELS


class ATruthsetHandler(ASampler):
    def __init__(self, granularity2source_file_dict, default_granularity, class_labels, random_state=None):

        self._default_granularity = default_granularity
        self._granularity2source_file_dict = granularity2source_file_dict
        self._class_labels = class_labels
        # set default random state
        self._random_state = random_state if random_state is not None else 2

        self._df_dict = {}

        for g, source_file in self._granularity2source_file_dict.items():
            self._load_df(g)

        #
        # prepare local cache
        self.__cache_n_value_for_balanced_classes = {}
        for g in self._granularity2source_file_dict:
            self.__cache_n_value_for_balanced_classes[g] = {}

    def _update_df(self, granularity, df):
        logging.debug("Update %s-level df" % granularity)
        self._df_dict[granularity] = df

    def _update_df_to_source_file(self, granularity, df):
        # ensure the same columns exist
        assert(df.columns == self.get_truthset(granularity).columns)
        # number of rows should stay equal
        assert(df.shape[0] == self.get_truthset(granularity).shape[0])

        logging.debug("Update df to file %s" % self._granularity2source_file_dict[granularity])
        df.to_csv(self._granularity2source_file_dict[granularity], index=False, sep=CSV_SEPARATOR, encoding='UTF-8')

        # refresh df
        logging.debug("Refresh df from df..")
        self._load_df(granularity, df)

    def _load_df(self, granularity, df=None):
        source_file = self._granularity2source_file_dict[granularity]
        if df is None:
            logging.debug("Load %s-level file %s" % (granularity, source_file))
            self._df_dict[granularity] = pd.read_csv(source_file, sep=CSV_SEPARATOR, index_col=None)
        else:
            logging.debug("Load %s-level df" % (granularity))
            self._df_dict[granularity] = df

        logging.debug("df shape: %s" % str(self._df_dict[granularity].shape))

        self._after_load_df()

    def _after_load_df(self):
        """ Override in case of filtering needs"""
        pass

    def get_source_file(self, granularity):
        return self._granularity2source_file_dict[granularity]

    def get_truthset(self, granularity=None):
        return self._df_dict[granularity]

    def get_default_truthset(self):
        g = self.get_default_granularity()
        return self._df_dict[g]

    def _get_n_value_for_balanced_classes(self, granularity, label_col):
        if label_col in self.__cache_n_value_for_balanced_classes[granularity]:
            return self.__cache_n_value_for_balanced_classes[granularity][label_col]

        max_values = []
        for c in self._class_labels:
            max_values.append(self._df_dict[granularity][self._df_dict[granularity][label_col] == c].shape[0])
        self.__cache_n_value_for_balanced_classes[granularity][label_col] = min(max_values)

        return self.__cache_n_value_for_balanced_classes[granularity][label_col]

    def get_granularity_levels(self):
        return self._granularity2source_file_dict.keys()

    def get_default_granularity(self):
        return self._default_granularity

    def get_classes(self):
        return self._class_labels

    def get_random_state(self):
        return self._random_state

    def get_balanced_sample_data_cfg(self,
                                     label_col,
                                     max_items_per_class=None,
                                     classes_to_filter=[],
                                     granularity=None,
                                     configure_numeric_label_col=False):

        if not granularity:
            granularity = self._default_granularity
            logging.debug("No granularity provided.. using default: %s" % granularity)

        label_col_numeric = "class_numeric"
        df = self.get_truthset(granularity)

        df = df[df[label_col].isin(classes_to_filter) == False]
        # select only where peers agreed
        # df = df[df[label_column + "Agreement"] != 1]

        classes = list(self._class_labels)  # create a copy

        # dataframe of items per label
        df_per_label = []

        if not max_items_per_class:
            # if not set, take the max number of items
            max_items_per_class = len(df)

        # assert(not len(data_cfg.labels) > 2)
        if (len(classes) > 1):
            for i in range(0, len(classes)):
                sub = df[df[label_col] == classes[i]].copy()
                sub[label_col_numeric] = i + 1
                df_per_label.append(sub)

        else:

            not_class_label = "Not" + classes[0]

            # df having label
            sub1 = df[df[label_col] == classes[0]]
            sub1[label_col_numeric] = 1
            df_per_label.append(sub1)

            #
            # df missing label
            df_non = df[df[label_col] != classes[0]]
            # override the 'other' labels with the special not_class_label
            df_non[label_col] = not_class_label  # pandas warning SettingWithCopyWarning can be ignored
            df_non[label_col_numeric] = 0  # pandas warning SettingWithCopyWarning can be ignored

            df_per_label.append(df_non)

            # update data_cfg: add the special not_class_label into the list of labels
            classes.append(not_class_label)

        #
        # consider max_items_per_class
        _df = None
        # _df_per_label = []
        # init d_target_length
        d_target_length = max_items_per_class
        for d in df_per_label:
            d_target_length = min([len(d), self._get_n_value_for_balanced_classes(granularity, label_col), max_items_per_class])

        for d in df_per_label:
            if _df is None:
                _d = d.sample(n=d_target_length, random_state=self._random_state).reset_index(drop=True)
                _df = _d
            else:
                _d = d.sample(n=d_target_length, random_state=self._random_state).reset_index(drop=True)
                _df = _df.append(_d)

            # _df_per_label.append(_d)

        # shuffle the final set
        _df = _df.sample(frac=1, random_state=self._random_state)

        #todo think about adding this kind of functionality somewhere else
        # if isinstance(self, ATwoGranularityTruthsetHandler): #if truthset provider has an textpartsselector functionality..
        #     df_s = self.get_truthset(GRANULARITY_SENTENCE)
        #     dw = DataWrapper(user_comments=UserComments(_df, TextPartsSelectorCached(df_s)))
        # else:
        dw = DataWrapper(user_comments=UserComments(_df))

        # create data configurator
        data_cfg = DataSampleConfig(
            label_column=label_col if not configure_numeric_label_col else label_col_numeric,
            classes=classes,
            max_items_per_class=max_items_per_class,
            data_wrapper=dw,
            # df_per_class=_df_per_label,  # dfs per label
            test_set=None
        )

        return data_cfg

class ASingleGranularityTruthsetHandler(ATruthsetHandler):
    def __init__(self, granularity, source_file, class_labels, random_state=2):

        super().__init__(granularity2source_file_dict={granularity : source_file},
                         default_granularity=granularity,
                         class_labels=class_labels,
                         random_state=random_state)

class ATwoGranularityTruthsetHandler(ATruthsetHandler):

    def __init__(self, default_granularity, comment_source_file, sentence_source_file, class_labels, random_state=2):

        assert(default_granularity in GRANULARITY_LEVELS)

        granularity2source_file_dict = {GRANULARITY_COMMENT: comment_source_file,
                                        GRANULARITY_SENTENCE : sentence_source_file}

        super().__init__(granularity2source_file_dict=granularity2source_file_dict,
                         default_granularity=default_granularity,
                         class_labels=class_labels,
                         random_state=random_state)

        df_s = self.get_truthset(GRANULARITY_SENTENCE)
        self._text_parts_selector_strategy = TextPartsSelectorCached(df_s)

    def get_text_parts_selector_strategy(self):
        return self._text_parts_selector_strategy