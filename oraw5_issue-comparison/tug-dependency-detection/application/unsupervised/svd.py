# -*- coding: utf-8 -*-

import logging
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.metrics import pairwise_distances
import warnings

_logger = logging.getLogger(__name__)


#===================================================================================================
# CONFIGURATION
#===================================================================================================

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        if self.key == "tokens":
            return np.array(map(lambda t: t[1], data_dict))

        if self.key == "direction":
            return [{
                'direction': 999 if direction else 0,
                #'rx': rx,
                #'ry': ry,
            } for (rx, ry, direction, _, _) in map(lambda t: t[0], data_dict)]

        tokens = []
        for (_, _, _, pos_tags_rx, pos_tags_ry) in map(lambda t: t[0], data_dict):
            tokens += [' '.join(pos_tags_rx + pos_tags_ry)]
        return np.array(tokens)


def extract_tokens(text): return text.split()

PARAMS_COUNT_VECTORIZER_COMMON = {
        'input': 'content',
        'tokenizer': extract_tokens,
        'preprocessor': None,
        'analyzer': 'word',
        'encoding': 'utf-8',
        'decode_error': 'strict',
        'strip_accents': None,
        'lowercase': True,
        'stop_words': None,
        'vocabulary': None,
        'binary': False
}


PARAMS_COUNT_VECTORIZER_NO_GRID_SEARCH = {
        #------------------------------------------------------------------------------------------
        # NOTE: parameters of best estimator determined by grid search go here:
        #------------------------------------------------------------------------------------------
        'max_features': 2000,
        'max_df': 0.85,
        'min_df': 2,
        'ngram_range': (1, 3),
}


DEFAULT_PARAMS_TFIDF_TRANSFORMER = {
        'norm': 'l2',
        'sublinear_tf': False,
        'smooth_idf': True,
        'use_idf': True
}

PARAMS_GRID_SEARCH_COMMON = {
#       # insert classifier parameters here!
#         'clf__alpha': [0.2, 0.1, 0.06, 0.03, 0.01, 0.001, 0.0001],
#        'clf__C': [2.0, 1.0]
}

PARAMS_GRID_SEARCH_TFIDF_FEATURES = {
        'featureselector__tokens__vectorizerTokens__max_features': (None, 400, 1000),
        'featureselector__tokens__vectorizerTokens__max_df': (0.4, 0.5, 0.7, 1.0),
        'featureselector__tokens__vectorizerTokens__min_df': (1, 2, 3),
        'featureselector__tokens__vectorizerTokens__ngram_range': ((1, 1), (1, 2), (1, 3), (1, 4)) # unigrams, bigrams or trigrams mixed
}
#===================================================================================================


def _evaluate(y_predicted_list, y_test_dependency_ids, X_test, y_test, requirements_map, use_numeric_features):
    # -----------------------------------------------------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------------------------------------------------
    y_predicted_label_list = []
    # print classifier.classes_
    n_true_positives = 0
    n_positives = 0
    n_true_negatives = 0
    n_negatives = 0
    for idx, _ in enumerate(y_predicted_list):
        rx_id, ry_id, dt = y_test_dependency_ids[idx].split("_")
        if y_predicted_list[idx] == True:
            n_positives += 1
            n_true_positives += 1 if y_test[idx] == y_predicted_list[idx] else 0
        else:
            n_negatives += 1
            n_true_negatives += 1 if y_test[idx] == y_predicted_list[idx] else 0

        #if y_test[idx] is True:
        #    print "{}: {} -> {} ({}, {})".format(y_test[idx] == y_predicted_list[idx], requirements_map[int(rx_id)],
        #                                         requirements_map[int(ry_id)], y_predicted_list[idx])

    y_predicted_fixed_size = np.array(y_predicted_list)

    # -----------------------------------------------------------------------------------------------
    # EVALUATION
    # -----------------------------------------------------------------------------------------------
    if not use_numeric_features:
        print("-" * 80)
        for item, labels in zip(X_test, y_predicted_label_list):
            print("{} -> ({})".format(item[:40], ', '.join(labels)))

    p_binary_true, r_binary_true, f1_binary_true, _ = precision_recall_fscore_support(y_test, y_predicted_fixed_size, pos_label=True,
                                                                       average="binary", warn_for=())

    print("#True positives: {}".format(n_true_positives))
    print("#Positives: {}".format(n_positives))
    print("#True negatives: {}".format( n_true_negatives))
    print("#Negatives: {}".format(n_negatives))
    print("Precision binary (true): {}".format(p_binary_true))

    print("Recall binary (true): {}".format(r_binary_true))
    print("F1 binary (true): {}".format(f1_binary_true))

    p_binary_false, r_binary_false, f1_binary_false, _ = precision_recall_fscore_support(y_test, y_predicted_fixed_size, pos_label=False,
                                                                                         average="binary", warn_for=())
    print("Precision binary (false): {}".format(p_binary_false))
    print("Recall binary (false): {}".format(r_binary_false))
    print("F1 binary (false): {}".format(f1_binary_false))

    p_binary_avg, r_binary_avg, f1_binary_avg, _ = precision_recall_fscore_support(y_test, y_predicted_fixed_size,
                                                                                   average="weighted", warn_for=())
    print("Precision binary (avg): {}".format(p_binary_avg))
    print("Recall binary (avg): {}".format(r_binary_avg))
    print("F1 binary (avg): {}".format(f1_binary_avg))

    print("Classification report:")
    print(metrics.classification_report(y_test, y_predicted_fixed_size))#, target_names=target_names)

    print("Confusion matrix:")
    print(metrics.confusion_matrix(y_test, y_predicted_fixed_size))


"""
def svd2(X_train, y_train, X_test, y_test, y_test_dependency_ids, requirements_map):
    parameters_count_vectorizer = PARAMS_COUNT_VECTORIZER_COMMON
    parameters_count_vectorizer = helper.merge_two_dicts(parameters_count_vectorizer,
                                                         PARAMS_COUNT_VECTORIZER_NO_GRID_SEARCH)
    svd_model = TruncatedSVD(n_components=500, algorithm='randomized', n_iter=10, random_state=42)
    svd_transformer = Pipeline([
        ('selector', ItemSelector(key='tokens')),
        ('vectorizerTokens', CountVectorizer(**parameters_count_vectorizer)),
        ('tfidfTokens', TfidfTransformer(**DEFAULT_PARAMS_TFIDF_TRANSFORMER)),
        ('svd_model', svd_model)
    ])
    svd_matrix = svd_transformer.fit_transform(X_train)
    from sklearn.metrics import pairwise_distances
    for idx, x_test in enumerate(X_test):
        # transform document into semantic space
        transformed_document = svd_transformer.transform(x_test)
        #print transformed_document
        distance_matrix = pairwise_distances(transformed_document, svd_matrix, metric='cosine', n_jobs=-1)
        print distance_matrix
        print '-' * 80
    return
"""


def svd(train_requirements, k=3, min_distance=0.0, max_distance=0.2):
    X_train = list(map(lambda r: ' '.join(map(lambda t: t.lower(), r.tokens(title_weight=1, description_weight=1))), train_requirements))

    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pandas", lineno=570)
    vectorizer = CountVectorizer(min_df=1) #TfidfVectorizer(min_df=1, ngram_range=(1,1))
    document_term_matrix = vectorizer.fit_transform(X_train)
    n_total_tokens = document_term_matrix.shape[1]
    n_components = min(int(n_total_tokens / 3), 300)

    if n_components == 0:
        return {}

    print("Desired components: {}".format(n_components))
    lsa = TruncatedSVD(n_components=n_components, algorithm='randomized', n_iter=300, random_state=1)
    document_term_matrix_lsa = lsa.fit_transform(document_term_matrix)
    print("Actual components: {}".format(document_term_matrix_lsa.shape[1]))
    #document_term_matrix_lsa = Normalizer(copy=False).fit_transform(document_term_matrix_lsa)

    predictions_for_requirements = {}
    for subject_requirement_idx, transfered_requirement in enumerate(document_term_matrix_lsa):
        transfered_requirement = np.array([transfered_requirement])
        distance_matrix = pairwise_distances(transfered_requirement, document_term_matrix_lsa, metric='cosine', n_jobs=1)

        p_similar_requirements = np.sort(distance_matrix[0])
        similar_train_requirement_idx = np.argsort(distance_matrix[0])

        n_recommended_dependencies = 0
        predictions = []
        subject_requirement = train_requirements[subject_requirement_idx]
        for inner_idx, similar_requirement_idx in enumerate(similar_train_requirement_idx):
            if subject_requirement_idx == similar_requirement_idx:
                continue

            if p_similar_requirements[inner_idx] < min_distance:
                continue

            if n_recommended_dependencies >= k or p_similar_requirements[inner_idx] > max_distance:
                break

            similar_requirement = train_requirements[similar_requirement_idx]
            n_recommended_dependencies += 1
            predictions += [similar_requirement]
        predictions_for_requirements[subject_requirement] = predictions

    return predictions_for_requirements
