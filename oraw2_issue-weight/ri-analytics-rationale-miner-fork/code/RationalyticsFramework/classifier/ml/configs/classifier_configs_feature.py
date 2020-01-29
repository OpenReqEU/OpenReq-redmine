"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of the Rationalytics framework and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

import logging

from sklearn.feature_extraction.text import TfidfVectorizer

# CONTEXT_FIRST_SENTENCE_SUFFIX = "_first_sentence"
CONTEXT_FIRST_SENTENCE_PREFIX = "first_sentence"
# CONTEXT_LAST_SENTENCE_SUFFIX = "_last_sentence"
CONTEXT_LAST_SENTENCE_PREFIX = "last_sentence"

# general algorithm_params
P_DATA_SLICE_NAME = "p_data_slice"
P_DATA_SELECTOR = "p_data_column_selector"
P_DATA_SELECTOR_PARAMS = "p_data_column_selector_params"

P_FEATURE_TYPE = "type"
P_PREPROCESSOR = "p_preprocesor"
P_PREPROCESSOR_PARAMS = "p_preprocesor_params"

P_FEATURE_EXTRACTOR = "p_feature_extractor"
P_FEATURE_EXTRACTOR_PARAMS = "p_feature_extractor_params"

P_FEATURE_NORMALIZER = "p_feature_normalizer"
P_FEATURE_NORMALIZER_PARAMS = "p_feature_normalizer_params"

P_FEATURE_DIM_REDUCER = "p_feature_dim_reducer"
P_FEATURE_DIM_REDUCER_PARAMS = "p_feature_dim_reducer_params"

#
# ngram feature type

F_TYPE_NGRAM = "f_ngram"
# for assertion checks
_P_NGRAM_PARAMS_REQ = [P_FEATURE_TYPE, P_DATA_SLICE_NAME, P_PREPROCESSOR, P_DATA_SELECTOR, P_FEATURE_EXTRACTOR]
_P_NGRAM_PARAMS_ALL = _P_NGRAM_PARAMS_REQ + [P_DATA_SELECTOR_PARAMS, P_PREPROCESSOR_PARAMS, P_FEATURE_EXTRACTOR_PARAMS, P_FEATURE_NORMALIZER, P_FEATURE_NORMALIZER_PARAMS, P_FEATURE_DIM_REDUCER, P_FEATURE_DIM_REDUCER_PARAMS]

#
# numeric feature type
F_TYPE_NUMERIC = "f_numeric"

#
# for assertion checks
_P_NUMERIC_REQ = [P_FEATURE_TYPE, P_DATA_SLICE_NAME, P_DATA_SELECTOR]
_P_NUMERIC_ALL = _P_NUMERIC_REQ + [P_PREPROCESSOR, P_PREPROCESSOR_PARAMS, P_DATA_SELECTOR_PARAMS, P_FEATURE_EXTRACTOR, P_FEATURE_EXTRACTOR_PARAMS, P_FEATURE_NORMALIZER, P_FEATURE_NORMALIZER_PARAMS, P_FEATURE_DIM_REDUCER, P_FEATURE_DIM_REDUCER_PARAMS]

#
# special keys feature type
F_SPECIAL_KEYS = "f_special_keys"
P_SPECIAL_KEYS_LIST = "p_special_keys_list"
#
# for assertion checks
_P_SPECIAL_KEYS_PARAMS_REQ = [P_FEATURE_TYPE, P_SPECIAL_KEYS_LIST]
_P_SPECIAL_KEYS_PARAMS_ALL = _P_SPECIAL_KEYS_PARAMS_REQ

#
# Assertion check. methods
#

def _check_params_validity(p_required_list, p_all_list, feature_params):
    _p_required = {k: False for k in p_required_list}

    for k, v in feature_params.items():
        if k not in p_all_list:
            logging.debug("feature id meta-type unknown: %s" % k)
            return False

        if k in _p_required.keys():
            _p_required[k] = True

    # True if all req. feature type params are included
    valid = False not in _p_required.values()
    if not valid:
        logging.debug("Missing required params: %s" % (",".join([k for k in _p_required if _p_required[k] == False])))

    return valid

def check_params_validity(feature_type, feature_params):
    if feature_type == F_TYPE_NGRAM:
        return check_ngram_params_validity(feature_params)
    elif feature_type == F_TYPE_NUMERIC:
        return check_numeric_params_validity(feature_params)

def check_ngram_params_validity(feature_params):
    return _check_params_validity(_P_NGRAM_PARAMS_REQ, _P_NGRAM_PARAMS_ALL, feature_params)

def check_numeric_params_validity(feature_params):
    return _check_params_validity(_P_NUMERIC_REQ, _P_NUMERIC_ALL, feature_params)

def check_feature_cfg(feature_type, feature_params):
    return check_params_validity(feature_type, feature_params)


# shortcut for evaluating bool values
def eval_dict_key_bool_value(key, **dict):
    return dict[key] if key in dict.keys() else False

#
# text features
F_TEXT = "text"
F_TEXT_POS = "text_pos"
F_TEXT_POS_CLAUSE = "text_pos_clause"
F_TEXT_POS_PHRASE = "text_pos_phrase"
F_TEXT_POS_CLAUSE_PHRASE = "text_pos_clausephrase"
F_TEXT_SENTIMENT_NORM = "text_sentiment_norm"
F_TEXT_LENGTH = "text_length"
F_TEXT_SYN_TREE_HEIGHT = "text_syn_tree_height"
F_TEXT_SYN_SUBTREE_COUNT = "text_syn_subtree_count"

F_TEXT_JMARKER_ANALOGY = "text_justification_marker_analogy"
F_TEXT_JMARKER_ANTITHESIS = "text_justification_marker_antithesis"
F_TEXT_JMARKER_CAUSE = "text_justification_marker_cause"
F_TEXT_JMARKER_CONCESSION = "text_justification_marker_concession"
F_TEXT_JMARKER_REASON = "text_justification_marker_reason"