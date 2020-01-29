"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of URMiner and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""


import logging

from RationalyticsFramework.classifier.ml.configs.classifier_configs_feature import F_TYPE_NUMERIC, F_TEXT
from RationalyticsFramework.classifier.data.preprocessors.preprocessor_basic import TextPreprocessor
from URMiner.classifier_ur.data import COL_TEXT_TITLE, COL_INDEX_SENTENCE
from URMiner.classifier_ur.data.data_provider_factory import URTruthsetHandlerFactory
from URMiner.classifier_ur.ml.classifier import AURClassifierConfig, UR_R_INDEX_SENTENCE, UR_F_RATING

class URReviewClassifierConfig(AURClassifierConfig):
    __th = URTruthsetHandlerFactory.ur_review()

    def __init__(self):
        super().__init__(self.__th)

    def init_granularity_spec_features(self):
        self._register_default_text_features(fid_prefix=COL_TEXT_TITLE, data_slice_name=COL_TEXT_TITLE)
        self._register_feature(id=UR_R_INDEX_SENTENCE,
                             data_slice_name=COL_INDEX_SENTENCE,
                             ftype=F_TYPE_NUMERIC)


class URBaselineReviewClassifierConfig(URReviewClassifierConfig):

    def _activate_features(self):
        logging.debug("activate feature cfg..")

        f_text_TITLE = self.get_title_fid(F_TEXT)
        self._activate_feature_id(f_text_TITLE)
        self.update_feature_preprocessor_params(f_text_TITLE, {TextPreprocessor.PARAM_NOPUNCT: True,
                                                            TextPreprocessor.PARAM_NOSTOPS: True,
                                                            TextPreprocessor.PARAM_DOLEMMATIZE: True})
        self.update_feature_extractor_params(f_text_TITLE, {"ngram_range": (1, 3)})

        # activate text-ngram feature id for body-text
        f_text_BODY = self.get_body_fid(F_TEXT)
        self._activate_feature_id(f_text_BODY)

        # update preprocessor parameter for text-ngram feature for body-text
        self.update_feature_preprocessor_params(f_text_BODY, {TextPreprocessor.PARAM_NOPUNCT: True,
                                                            TextPreprocessor.PARAM_NOSTOPS: True,
                                                            TextPreprocessor.PARAM_DOLEMMATIZE: True})

                                                            
class URNewReviewClassifierConfig(URReviewClassifierConfig):

    def _activate_features(self):
        logging.debug("activate feature cfg..")

        f_text_TITLE = self.get_title_fid(F_TEXT)
        self._activate_feature_id(f_text_TITLE)
        self.update_feature_preprocessor_params(f_text_TITLE, {TextPreprocessor.PARAM_NOPUNCT: True,
                                                            TextPreprocessor.PARAM_NOSTOPS: True,
                                                            TextPreprocessor.PARAM_DOLEMMATIZE: True})
        self.update_feature_extractor_params(f_text_TITLE, {"ngram_range": (1, 3)})

        # activate text-ngram feature id for body-text
        f_text_BODY = self.get_body_fid(F_TEXT)
        self._activate_feature_id(f_text_BODY)

        # update preprocessor parameter for text-ngram feature for body-text
        self.update_feature_preprocessor_params(f_text_BODY, {TextPreprocessor.PARAM_NOPUNCT: True,
                                                            TextPreprocessor.PARAM_NOSTOPS: True,
                                                            TextPreprocessor.PARAM_DOLEMMATIZE: True})

        # self._activate_feature_id(UR_F_RATING)