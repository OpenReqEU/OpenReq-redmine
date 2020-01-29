"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of the Rationalytics framework and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

import copy
import logging
from abc import ABC, abstractmethod

from RationalyticsFramework.classifier.ml.configs.classifier_configs_algorithm import ALGO_CODE_NAIVEB, ALGO_CODE_SUPPORT_VC, \
    ALGO_CODE_LOGISTIC_REGRESSION, ALGO_CODE_DECISION_TREE, ALGO_CODE_RANDOM_FOREST, ALGO_CODE_GAUSSIAN_PROCESS, \
    ALGO_CODE_MULTILAYER_PERCEPTRON_CLF, get_classifier_algorithm_config
from RationalyticsFramework.classifier.data.data_selectors.data_selector_basic import DataPreSelector  # , FloatDataSelector
from RationalyticsFramework.classifier.data.preprocessors.preprocessor_basic import TextPreprocessor, TextLengthExtractor, APreprocessor
from RationalyticsFramework.classifier.data.preprocessors.preprocessor_marker import AnalogyMarker, AntiThesisMarker, CauseMarker, \
    ConcessionMarker, ReasonMarker
from RationalyticsFramework.classifier.data.preprocessors.preprocessor_sentiment import SentimentExtractor
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

from RationalyticsFramework.classifier.data.preprocessors.preprocessor_syntags import POSTagsExtractor, PENNClauseTagsExtractor, \
    PENNPhraseTagsExtractor, PENNClausePhraseTagsExtractor, SynTreeHeightExtractor, SynSubTreeCountExtractor
from RationalyticsFramework.classifier.ml.configs.classifier_configs_feature import P_DATA_SLICE_NAME, F_TYPE_NGRAM, P_FEATURE_TYPE, \
    P_DATA_SELECTOR, \
    F_TYPE_NUMERIC, check_feature_cfg, P_PREPROCESSOR, \
    P_PREPROCESSOR_PARAMS, P_DATA_SELECTOR_PARAMS, P_FEATURE_NORMALIZER, \
    P_FEATURE_NORMALIZER_PARAMS, P_FEATURE_DIM_REDUCER, P_FEATURE_DIM_REDUCER_PARAMS, P_FEATURE_EXTRACTOR, \
    P_FEATURE_EXTRACTOR_PARAMS, F_TEXT, F_TEXT_POS, F_TEXT_POS_CLAUSE, F_TEXT_POS_PHRASE, F_TEXT_SENTIMENT_NORM, \
    F_TEXT_LENGTH, F_TEXT_SYN_TREE_HEIGHT, F_TEXT_SYN_SUBTREE_COUNT, F_TEXT_POS_CLAUSE_PHRASE, F_TEXT_JMARKER_ANALOGY, \
    F_TEXT_JMARKER_ANTITHESIS, F_TEXT_JMARKER_CAUSE, F_TEXT_JMARKER_CONCESSION, F_TEXT_JMARKER_REASON


class AClassifierConfig(ABC):

    def __init__(self, truthset_handler):
        self._truthset_handler = truthset_handler

        self._feature_id_list = []
        self._feature_cfg_activated_dict = {}
        self._feature_cfg_dict = {}

        self._default_classifier_algorithm = None

        # init classifier algorithm (optional)
        self._init_default_classifier_algorithm()

        # init features
        self._register_features()

        # init features
        self._activate_features()

        # ensure the feature id list and cfg dict are set
        assert(self._feature_id_list and self._feature_cfg_dict and self._feature_cfg_activated_dict)


    def _register_feature(self, id, data_slice_name, ftype,
                          preprocessor=None, preprocessor_params=None,
                          selector=None, selector_params=None,
                          extractor=None, extractor_params=None,
                          normalizer=None, normalizer_params=None,
                          dim_reducer=None, dim_reducer_params=None):

        assert ftype in [F_TYPE_NGRAM, F_TYPE_NUMERIC]

        assert not (not preprocessor and preprocessor_params)
        assert not (not selector and selector_params)
        assert not (not extractor and extractor_params)
        assert not (not normalizer and normalizer_params)
        assert not (not dim_reducer and dim_reducer_params)

        # id = self._create_compound_id(id=id, feature_id=id)
        logging.debug("Import feature %s" % id)

        # create new dict for this feature
        # self._feature_cfg_dict[id] = {}
        d = {}

        d[P_FEATURE_TYPE] = ftype
        d[P_DATA_SLICE_NAME] = data_slice_name

        #
        # Selector (REQUIRED)
        d[P_DATA_SELECTOR] = selector if selector else DataPreSelector
        d[P_DATA_SELECTOR_PARAMS] = selector_params if selector_params else {
            DataPreSelector.CTOR_P_KEYS: None}    #None means all data keys

        logging.debug(
            "Set selector %s with params %s" % (d[P_DATA_SELECTOR], d[P_DATA_SELECTOR_PARAMS]))

        #
        # Preprocessor (OPTIONAL)
        if preprocessor:
            d[P_PREPROCESSOR] = preprocessor
            d[P_PREPROCESSOR_PARAMS] = preprocessor_params if preprocessor_params else \
                {APreprocessor.CTOR_PARAM_COL_PREFIX: data_slice_name}
            logging.debug("Set preprocessor %s with params %s" % (preprocessor, d[P_PREPROCESSOR_PARAMS]))


        #
        # Extractor (OPTIONAL)
        if extractor:
            d[P_FEATURE_EXTRACTOR] = extractor
            d[P_FEATURE_EXTRACTOR_PARAMS] = extractor_params if extractor_params else {}

            logging.debug(
                "Set extractor %s with params %s" % (
                    d[P_FEATURE_EXTRACTOR],
                    d[P_FEATURE_EXTRACTOR_PARAMS]))
        else:
            if ftype == F_TYPE_NGRAM:
                d[P_FEATURE_EXTRACTOR] = extractor if extractor else CountVectorizer
                d[P_FEATURE_EXTRACTOR_PARAMS] = extractor_params if extractor_params else {
                    "ngram_range": (1, 1), "binary": True}

                logging.debug(
                    "Set default extractor %s with params %s" % (
                    d[P_FEATURE_EXTRACTOR], d[P_FEATURE_EXTRACTOR_PARAMS]))

        #
        # Normalizer (OPTIONAL)
        if normalizer:
            d[P_FEATURE_NORMALIZER] = normalizer
            d[P_FEATURE_NORMALIZER_PARAMS] = normalizer_params if normalizer_params else {}
            logging.debug(
                "Set normalizer %s with params %s" % (
                    d[P_FEATURE_NORMALIZER],
                    d[P_FEATURE_NORMALIZER_PARAMS]))
        else:
            if ftype == F_TYPE_NGRAM:
                d[P_FEATURE_NORMALIZER] = normalizer if normalizer else TfidfTransformer
                d[P_FEATURE_NORMALIZER_PARAMS] = {}
                logging.debug(
                    "Set default normalizer %s with params %s" % (
                        d[P_FEATURE_NORMALIZER],
                        d[P_FEATURE_NORMALIZER_PARAMS]))
        #
        # Dim. reducer (OPTIONAL)
        if dim_reducer:
            d[P_FEATURE_DIM_REDUCER] = dim_reducer
            d[P_FEATURE_DIM_REDUCER] = dim_reducer_params
            logging.debug(
                "Set normalizer %s with params %s" % (dim_reducer, dim_reducer_params))

        logging.debug("assertion check feature of type %s: %s" % (ftype, d))
        assert (check_feature_cfg(ftype, d))

        self._feature_cfg_dict[id] = copy.deepcopy(d) # ensure the dict is copied


    def _register_default_text_features(self, fid_prefix, data_slice_name,
                                        ftype2data_selector_dict=None,
                                        ftype2data_selector_param_dict=None):

        if ftype2data_selector_dict is None:
            ftype2data_selector_dict = {F_TYPE_NGRAM: None, F_TYPE_NUMERIC: None}
        if ftype2data_selector_param_dict is None:
            ftype2data_selector_param_dict = {F_TYPE_NGRAM: None, F_TYPE_NUMERIC: None}

        self._register_feature(id=self._create_compound_id(F_TEXT, fid_prefix), data_slice_name=data_slice_name, ftype=F_TYPE_NGRAM,
                               preprocessor=TextPreprocessor, selector=ftype2data_selector_dict[F_TYPE_NGRAM], selector_params=ftype2data_selector_param_dict[F_TYPE_NGRAM])
        self._register_feature(id=self._create_compound_id(F_TEXT_POS, fid_prefix), data_slice_name=data_slice_name, ftype=F_TYPE_NGRAM,
                               preprocessor=POSTagsExtractor, selector=ftype2data_selector_dict[F_TYPE_NGRAM], selector_params=ftype2data_selector_param_dict[F_TYPE_NGRAM])
        self._register_feature(id=self._create_compound_id(F_TEXT_POS_CLAUSE, fid_prefix), data_slice_name=data_slice_name, ftype=F_TYPE_NGRAM,
                               preprocessor=PENNClauseTagsExtractor, selector=ftype2data_selector_dict[F_TYPE_NGRAM], selector_params=ftype2data_selector_param_dict[F_TYPE_NGRAM])
        self._register_feature(id=self._create_compound_id(F_TEXT_POS_PHRASE, fid_prefix), data_slice_name=data_slice_name, ftype=F_TYPE_NGRAM,
                               preprocessor=PENNPhraseTagsExtractor, selector=ftype2data_selector_dict[F_TYPE_NGRAM], selector_params=ftype2data_selector_param_dict[F_TYPE_NGRAM])
        self._register_feature(id=self._create_compound_id(F_TEXT_POS_CLAUSE_PHRASE, fid_prefix), data_slice_name=data_slice_name, ftype=F_TYPE_NGRAM,
                               preprocessor=PENNClausePhraseTagsExtractor, selector=ftype2data_selector_dict[F_TYPE_NGRAM], selector_params=ftype2data_selector_param_dict[F_TYPE_NGRAM])
        self._register_feature(id=self._create_compound_id(F_TEXT_SENTIMENT_NORM, fid_prefix), data_slice_name=data_slice_name, ftype=F_TYPE_NUMERIC,
                               preprocessor=SentimentExtractor, selector=ftype2data_selector_dict[F_TYPE_NUMERIC], selector_params=ftype2data_selector_param_dict[F_TYPE_NUMERIC])
        self._register_feature(id=self._create_compound_id(F_TEXT_LENGTH, fid_prefix), data_slice_name=data_slice_name, ftype=F_TYPE_NUMERIC,
                               preprocessor=TextLengthExtractor, selector=ftype2data_selector_dict[F_TYPE_NUMERIC], selector_params=ftype2data_selector_param_dict[F_TYPE_NUMERIC])
        self._register_feature(id=self._create_compound_id(F_TEXT_SYN_TREE_HEIGHT, fid_prefix), data_slice_name=data_slice_name, ftype=F_TYPE_NUMERIC,
                               preprocessor=SynTreeHeightExtractor, selector=ftype2data_selector_dict[F_TYPE_NUMERIC], selector_params=ftype2data_selector_param_dict[F_TYPE_NUMERIC])
        self._register_feature(id=self._create_compound_id(F_TEXT_SYN_SUBTREE_COUNT, fid_prefix), data_slice_name=data_slice_name, ftype=F_TYPE_NUMERIC,
                               preprocessor=SynSubTreeCountExtractor, selector=ftype2data_selector_dict[F_TYPE_NUMERIC], selector_params=ftype2data_selector_param_dict[F_TYPE_NUMERIC])

        self._register_feature(id=self._create_compound_id(F_TEXT_JMARKER_ANALOGY, fid_prefix), data_slice_name=data_slice_name, ftype=F_TYPE_NUMERIC,
                               preprocessor=AnalogyMarker, selector=ftype2data_selector_dict[F_TYPE_NUMERIC], selector_params=ftype2data_selector_param_dict[F_TYPE_NUMERIC])
        self._register_feature(id=self._create_compound_id(F_TEXT_JMARKER_ANTITHESIS, fid_prefix), data_slice_name=data_slice_name, ftype=F_TYPE_NUMERIC,
                               preprocessor=AntiThesisMarker, selector=ftype2data_selector_dict[F_TYPE_NUMERIC], selector_params=ftype2data_selector_param_dict[F_TYPE_NUMERIC])
        self._register_feature(id=self._create_compound_id(F_TEXT_JMARKER_CAUSE, fid_prefix), data_slice_name=data_slice_name, ftype=F_TYPE_NUMERIC,
                               preprocessor=CauseMarker, selector=ftype2data_selector_dict[F_TYPE_NUMERIC], selector_params=ftype2data_selector_param_dict[F_TYPE_NUMERIC])
        self._register_feature(id=self._create_compound_id(F_TEXT_JMARKER_CONCESSION, fid_prefix),
                               data_slice_name=data_slice_name, ftype=F_TYPE_NUMERIC,
                               preprocessor=ConcessionMarker, selector=ftype2data_selector_dict[F_TYPE_NUMERIC],
                               selector_params=ftype2data_selector_param_dict[F_TYPE_NUMERIC])
        self._register_feature(id=self._create_compound_id(F_TEXT_JMARKER_REASON, fid_prefix),
                               data_slice_name=data_slice_name, ftype=F_TYPE_NUMERIC,
                               preprocessor=ReasonMarker, selector=ftype2data_selector_dict[F_TYPE_NUMERIC],
                               selector_params=ftype2data_selector_param_dict[F_TYPE_NUMERIC])

    def _create_compound_id(self, id, id_prefix):
        if id_prefix:
            return "%s_%s" % (id_prefix, id)

        return id

    @abstractmethod
    def _init_default_classifier_algorithm(self):
        pass

    @abstractmethod
    def _register_features(self):
        pass

    @abstractmethod
    def _activate_features(self):
        pass

    def set_default_classifier_algorithm_via_code(self, algo_code):
        algo_cfg = get_classifier_algorithm_config(algo_code)
        self._default_classifier_algorithm = algo_cfg

    def set_default_classifier_algorithm(self, algo):
        self._default_classifier_algorithm = algo

    def get_default_classifier_algorithm(self):
        return self._default_classifier_algorithm

    def _activate_feature_id(self, f_id):
        if f_id in self._feature_id_list:
            raise Exception("fid already in feature id list")

        logging.debug("Activating feature %s" % f_id)

        self._feature_id_list.append(f_id)

        # update feature cfg
        self._feature_cfg_activated_dict = {key: self._feature_cfg_dict[key] for key in self._feature_id_list}

    def get_truthset_handler(self):
        return self._truthset_handler

    def get_default_granularity(self):
        return self._truthset_handler.get_default_granularity()

    def get_feature_cfg_id_list(self):
        return self._feature_id_list

    def update_feature_config_param_dict(self, feature_id, param, value):
        assert(isinstance(value, dict))
        logging.debug("update feature cfg %s param %s with value %s" % (feature_id, param, value))

        cid = self.get_feature_unique_id(feature_id)
        self._feature_cfg_activated_dict[cid][param].update(value)

        logging.debug("resulting feature cfg %s param %s:" % (feature_id, self._feature_cfg_activated_dict[cid][param]))

    def set_feature_config_param(self, feature_id, param, value):
        cid = self.get_feature_unique_id(feature_id)
        self._feature_cfg_activated_dict[cid][param] = value

    def get_feature_config_param(self, feature_id, param):
        cid = self.get_feature_unique_id(feature_id)

        if param not in self._feature_cfg_activated_dict[cid]:
            logging.debug("Param %s not found in feature_cfg_dict" % param)
            return None

        return self._feature_cfg_dict[cid][param]

    def get_activated_features_cfg_dict(self):
        return self._feature_cfg_activated_dict

    def get_feature_config_dict_params(self, feature_id):
        cid = self.get_feature_unique_id(feature_id)
        return self._feature_cfg_activated_dict[cid]

    #todo no need for this method any more
    def get_feature_unique_id(self, feature_id):
        return feature_id

    def get_classifier_algorithm_config(self, code):
        return get_classifier_algorithm_config(code)

    def get_feature_data_column(self, feature_id):
        p = self.get_feature_config_param(feature_id, P_DATA_SLICE_NAME)
        return p

    def get_feature_type(self, feature_id):
        return self._feature_cfg_dict[feature_id][P_FEATURE_TYPE]

    #
    # Preprocessor
    def get_feature_preprocessor(self, feature_id):
        p = self.get_feature_config_param(feature_id, P_PREPROCESSOR)
        return p

    def get_feature_preprocessor_params(self, feature_id):
        p = self.get_feature_config_param(feature_id, P_PREPROCESSOR_PARAMS)
        return p

    def update_feature_preprocessor_params(self, feature_id, params):
        self.update_feature_config_param_dict(feature_id, P_PREPROCESSOR_PARAMS, params)

    #
    # item selector
    def get_item_selector(self, feature_id):
        p = self.get_feature_config_param(feature_id, P_DATA_SELECTOR)
        return p

    def get_item_selector_params(self, feature_id):
        p = self.get_feature_config_param(feature_id, P_DATA_SELECTOR_PARAMS)
        return p

    def update_item_selector_params(self, feature_id, params):
        self.update_feature_config_param_dict(feature_id, P_DATA_SELECTOR_PARAMS, params)

    #
    # feature extractor
    def get_feature_extractor(self, feature_id):
        n = self.get_feature_config_param(feature_id, P_FEATURE_EXTRACTOR)
        return n

    def get_feature_extractor_params(self, feature_id):
        n = self.get_feature_config_param(feature_id, P_FEATURE_EXTRACTOR_PARAMS)
        return n

    def update_feature_extractor_params(self, feature_id, params):
        self.update_feature_config_param_dict(feature_id, P_FEATURE_EXTRACTOR_PARAMS, params)

    #
    # feature normalizer
    def set_feature_normalizer(self, feature_id, normalizer_class):
        n = self.set_feature_config_param(feature_id, P_FEATURE_NORMALIZER, normalizer_class)

    def get_feature_normalizer(self, feature_id):
        n = self.get_feature_config_param(feature_id, P_FEATURE_NORMALIZER)
        return n

    def get_feature_normalizer_params(self, feature_id):
        n = self.get_feature_config_param(feature_id, P_FEATURE_NORMALIZER_PARAMS)
        return n

    def update_feature_normalizer_params(self, feature_id, params):
        self.update_feature_config_param_dict(feature_id, P_FEATURE_NORMALIZER_PARAMS, params)

    #
    # feature dim reducer
    def get_feature_dim_reducer(self, feature_id):
        n = self.get_feature_config_param(feature_id, P_FEATURE_DIM_REDUCER)
        return n

    def get_feature_dim_reducer_params(self, feature_id):
        n = self.get_feature_config_param(feature_id, P_FEATURE_DIM_REDUCER_PARAMS)
        return n

    def update_feature_dim_reducer_params(self, feature_id, params):
        self.update_feature_config_param_dict(feature_id, P_FEATURE_DIM_REDUCER_PARAMS, params)

    def get_data_sample_config(self, **params):
        #todo allow supplyment of granularity
        return self._truthset_handler.get_balanced_sample_data_cfg(**params)

def get7classifier_codes():
    return [ALGO_CODE_NAIVEB, ALGO_CODE_SUPPORT_VC, ALGO_CODE_LOGISTIC_REGRESSION, ALGO_CODE_DECISION_TREE,
            ALGO_CODE_RANDOM_FOREST,
            ALGO_CODE_GAUSSIAN_PROCESS, ALGO_CODE_MULTILAYER_PERCEPTRON_CLF]

def get_baseline_classifier_algorithm_codes():
    clf_codes = get7classifier_codes()
    return clf_codes

def get_baseline_classifier_algorithm_configs():
    clf_codes = get_baseline_classifier_algorithm_codes()
    clf_configs4baseline = [get_classifier_algorithm_config(clf_code) for clf_code in clf_codes]

    return clf_configs4baseline