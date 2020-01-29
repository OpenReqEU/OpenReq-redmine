"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of URMiner and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""


import logging

# from configs_ur.classifier import URBaselineSentenceClassifierConfig
# from configs_ur.classifier_sentence import URBaselineSentenceClassifierConfig
from pandas import DataFrame

from RationalyticsFramework.classifier.ml.rationalytics_classifier import ARationalyticsClassifier
from URMiner.classifier_ur.data import COL_TEXT_TITLE, COL_RATING, COL_TEXT_BODY
from URMiner.consts_ur import ROOT_DIR
from RationalyticsFramework.classifier.data.data_wrapper import DataWrapper, UserComments

class URClassifier(ARationalyticsClassifier):
    MODEL_FOLDER = ROOT_DIR + "/services_ur/classifier_models/"

    def __init__(self, classifier_cfg, label_column, tag):
        super().__init__(self.MODEL_FOLDER, classifier_cfg, label_column, tag)

        self._label_column = label_column
        self._classifier_cfg = classifier_cfg
        self._classifier_algo_cfg = classifier_cfg.get_default_classifier_algorithm()
        self._base_filename = "%s%s_%s_%s" % (self.MODEL_FOLDER, self._label_column, self._classifier_algo_cfg.algorithm_code, tag)
        self._model = None

    def predict(self, d):
        return self.predict_ur(Title=d[COL_TEXT_TITLE],
                       Body=d[COL_TEXT_BODY],
                       Rating=d[COL_RATING])

    def predict_list(self, dlist):
        out = []
        for d in dlist:
            r = self.predict_ur(Title=d[COL_TEXT_TITLE],
                           Body=d[COL_TEXT_BODY],
                           Rating=d[COL_RATING])

            out.append(r)

        return out

    def predict_dict(self, d):

        item_df = DataFrame(d, index=[0])

        if not self._model:
            self.train()

        logging.debug("classes:" + str(self._model.classes_))
        data_wrapper = DataWrapper(user_comments=UserComments(item_df))
        df = data_wrapper.compiled_df()

        res = self._model.predict_proba(df)
        res_dict = {}
        for c_i, c in enumerate(self._model.classes_):
            res_dict[str(c)] = res[0][c_i]

        return res_dict


    def predict_ur(self, Title, Body, Rating):
        item_dict = {
                COL_TEXT_TITLE : Title,
                COL_TEXT_BODY: Body,
                COL_RATING: Rating}

        item_df = DataFrame(item_dict, index=[0])

        if not self._model:
            self.train()

        logging.debug("classes:" + str(self._model.classes_))
        data_wrapper = DataWrapper(user_comments=UserComments(item_df))
        df = data_wrapper.compiled_df()

        res = self._model.predict_proba(df)
        res_dict = {}
        for c_i, c in enumerate(self._model.classes_):
            res_dict[str(c)] = res[0][c_i]

        return res_dict
