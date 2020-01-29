"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of URMiner and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""


from abc import abstractmethod

from RationalyticsFramework.classifier.ml.configs.classifier_configs import AClassifierConfig
from RationalyticsFramework.classifier.ml.configs.classifier_configs_feature import F_TYPE_NUMERIC, F_TEXT, F_TEXT_POS, F_TEXT_POS_CLAUSE, \
    F_TEXT_POS_PHRASE, F_TEXT_SENTIMENT_NORM, \
    F_TEXT_SYN_TREE_HEIGHT, F_TEXT_SYN_SUBTREE_COUNT
from RationalyticsFramework.classifier.data.preprocessors.preprocessor_basic import TextPreprocessor
from URMiner.classifier_ur.data import COL_RATING, COL_TEXT_BODY, COL_TEXT_TITLE
from RationalyticsFramework.classifier.data import GRANULARITY_COMMENT

#
# custom
UR_F_RATING = "rating"
UR_R_INDEX_SENTENCE = "index_sentence"

class AURClassifierConfig(AClassifierConfig):
    def __init__(self, truthset_handler):
        super().__init__(truthset_handler)

    def _init_default_classifier_algorithm(self):
        pass

    def _register_features(self):
        self._register_default_text_features(fid_prefix=COL_TEXT_BODY, data_slice_name=COL_TEXT_BODY)

        self._register_feature(id=UR_F_RATING,
                             data_slice_name=COL_RATING,
                             ftype=F_TYPE_NUMERIC)

        # finally, init granularity spec. features
        self.init_granularity_spec_features()

    def get_body_fid(self, f_id):
        # create a new feature id of body-text by prefixing f_id
        f_id = self._create_compound_id(f_id, COL_TEXT_BODY)
        return f_id

    def get_title_fid(self, f_id):
        # create a compound feature id for title-text by prefixing f_id
        f_id = self._create_compound_id(f_id, COL_TEXT_TITLE)
        return f_id

    @abstractmethod
    def init_granularity_spec_features(self):
        pass

#
# Allfeaturetypes configs
#

class URAllFeatureTypesClassifierConfig(AURClassifierConfig):
    def __init__(self, truthset_handler):
        super().__init__(truthset_handler)

    def _activate_features(self):

        # prepare feature ids
        granularity = self.get_default_granularity()

        if granularity == GRANULARITY_COMMENT:
            f_title_text = self.get_title_fid(F_TEXT)
            self._activate_feature_id(f_title_text)
            self.update_feature_preprocessor_params(f_title_text, {TextPreprocessor.PARAM_NOPUNCT: True,
                                                                 TextPreprocessor.PARAM_NOSTOPS: True,
                                                                 TextPreprocessor.PARAM_DOLEMMATIZE: True})

            self.update_feature_extractor_params(f_title_text, {"ngram_range": (1, 3)})

        self._activate_feature_id(self.get_body_fid(F_TEXT))
        self._activate_feature_id(self.get_body_fid(F_TEXT_POS))
        self._activate_feature_id(self.get_body_fid(F_TEXT_POS_CLAUSE))
        self._activate_feature_id(self.get_body_fid(F_TEXT_POS_PHRASE))
        self._activate_feature_id(self.get_body_fid(F_TEXT_SENTIMENT_NORM))
        self._activate_feature_id(self.get_body_fid(F_TEXT_SYN_TREE_HEIGHT))
        self._activate_feature_id(self.get_body_fid(F_TEXT_SYN_SUBTREE_COUNT))

        self.update_feature_preprocessor_params(self.get_body_fid(F_TEXT), {TextPreprocessor.PARAM_NOPUNCT: True,
                                                            TextPreprocessor.PARAM_NOSTOPS: True,
                                                            TextPreprocessor.PARAM_DOLEMMATIZE: True})

        self.update_feature_extractor_params(self.get_body_fid(F_TEXT), {"ngram_range": (1, 3)})
        self.update_feature_extractor_params(self.get_body_fid(F_TEXT_POS), {"ngram_range": (1, 3)})
        self.update_feature_extractor_params(self.get_body_fid(F_TEXT_POS_CLAUSE), {"ngram_range": (1, 1), "binary" : False})
        self.update_feature_extractor_params(self.get_body_fid(F_TEXT_POS_PHRASE), {"ngram_range": (1, 1), "binary" : False})

#
# For Debug
#

class D_URFeatureTypesClassifierConfig(AURClassifierConfig):
    def __init__(self, truthset_handler):
        super().__init__(truthset_handler)

    def _activate_features(self):

        # prepare feature ids
        granularity = self.get_default_granularity()

        if granularity == GRANULARITY_COMMENT:
            f_title_text = self.get_title_fid(F_TEXT)
            self._activate_feature_id(f_title_text)
            self.update_feature_preprocessor_params(f_title_text, {TextPreprocessor.PARAM_NOPUNCT: True,
                                                                 TextPreprocessor.PARAM_NOSTOPS: True,
                                                                 TextPreprocessor.PARAM_DOLEMMATIZE: True})

            self.update_feature_extractor_params(f_title_text, {"ngram_range": (1, 3)})

        self._activate_feature_id(self.get_body_fid(F_TEXT))
        self._activate_feature_id(self.get_body_fid(F_TEXT_POS))
        self._activate_feature_id(self.get_body_fid(F_TEXT_POS_CLAUSE))
        self._activate_feature_id(self.get_body_fid(F_TEXT_POS_PHRASE))
        self._activate_feature_id(self.get_body_fid(F_TEXT_SENTIMENT_NORM))
        self._activate_feature_id(self.get_body_fid(F_TEXT_SYN_TREE_HEIGHT))
        self._activate_feature_id(self.get_body_fid(F_TEXT_SYN_SUBTREE_COUNT))

        self.update_feature_preprocessor_params(self.get_body_fid(F_TEXT), {
                                                            TextPreprocessor.PARAM_NOPUNCT: True,
                                                            TextPreprocessor.PARAM_NOSTOPS: True,
                                                            TextPreprocessor.PARAM_DOLEMMATIZE: True})

        self.update_feature_extractor_params(self.get_body_fid(F_TEXT), {"ngram_range": (1, 3)})
        self.update_feature_extractor_params(self.get_body_fid(F_TEXT_POS), {"ngram_range": (1, 3)})
        self.update_feature_extractor_params(self.get_body_fid(F_TEXT_POS_CLAUSE), {"ngram_range": (1, 1), "binary" : False})
        self.update_feature_extractor_params(self.get_body_fid(F_TEXT_POS_PHRASE), {"ngram_range": (1, 1), "binary" : False})

