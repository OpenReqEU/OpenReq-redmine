"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of the Rationalytics framework and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

import collections

#
# Classifier Configurations
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier, BernoulliRBM, MLPRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

CONFIG_CLASSIFIER_IDS = ["algorithm_code", "algorithm_instance", "algorithm_params", "dense_matrix_required"]
ClassifierAlgorithmConfig  = collections.namedtuple("ClassifierAlgorithmConfig",
                                                    [
                            CONFIG_CLASSIFIER_IDS[0],
                            CONFIG_CLASSIFIER_IDS[1],
                            CONFIG_CLASSIFIER_IDS[2],
                            CONFIG_CLASSIFIER_IDS[3]
                         ])

def _ger_algorithm_default_params(algorithm_code):
    if algorithm_code == "svc":
        return {"kernel" : "linear"}
    else:
        return {}


ALGO_CODE_DUMMY = "dummy"
ALGO_CODE_NAIVEB = "nb"
ALGO_CODE_SUPPORT_VC = "svc"
ALGO_CODE_DECISION_TREE = "dt"
ALGO_CODE_LOGISTIC_REGRESSION = "lr"
ALGO_CODE_GAUSSIAN_PROCESS = "gpc"
ALGO_CODE_RANDOM_FOREST = "rf"
ALGO_CODE_EXTRA_TREE = "etree"
ALGO_CODE_MULTILAYER_PERCEPTRON_CLF = "mpc"
ALGO_CODE_MULTILAYER_PERCEPTRON_REG = "mpr"
ALGO_CODE_BERNOULLI_RBM = "rbm"
ALGO_CODE_QUADRATIC_DA = "qda"

ALGO_CODE_NAME_MAPPING = {
           ALGO_CODE_DUMMY: "DUMMY 4Debug Classifier",
           ALGO_CODE_NAIVEB: "Naive Bayes",
           ALGO_CODE_SUPPORT_VC: "Support Vector Classifier",
           ALGO_CODE_DECISION_TREE: "Decision Tree",
           ALGO_CODE_LOGISTIC_REGRESSION: "Logistic Regression",
           ALGO_CODE_GAUSSIAN_PROCESS: "Gaussian Process Classifier",
           ALGO_CODE_RANDOM_FOREST: "Random Forest",
           ALGO_CODE_EXTRA_TREE: "Extra Tree Classifier",
           ALGO_CODE_MULTILAYER_PERCEPTRON_CLF: "Multi-layer Perceptron Classifier",
           ALGO_CODE_MULTILAYER_PERCEPTRON_REG: "Multi-layer Perceptron Regressor",
           ALGO_CODE_BERNOULLI_RBM: "Bernoulli Restricted Boltzmann Machine",
           ALGO_CODE_QUADRATIC_DA: "Quadratic Discriminant Analysis"}

ALGO_CODE_REQUIRING_DENSE_MATRIX=[ALGO_CODE_GAUSSIAN_PROCESS, ALGO_CODE_QUADRATIC_DA]

def get_algo_name(algorithm_code):
    return ALGO_CODE_NAME_MAPPING[algorithm_code]

def get_all_algo_codes():
    return ALGO_CODE_NAME_MAPPING.keys()

DUMMY_CLF_RANDOM_STATE = 2
def _get_algorithm_instance(algorithm_code, params):
    if algorithm_code == ALGO_CODE_DUMMY:
        algorithm = DummyClassifier(random_state=DUMMY_CLF_RANDOM_STATE, **params)
    elif algorithm_code == ALGO_CODE_NAIVEB:
        algorithm = MultinomialNB(**params)
    elif algorithm_code == ALGO_CODE_SUPPORT_VC:
        algorithm = SVC(**params)
    elif algorithm_code == ALGO_CODE_DECISION_TREE:
        algorithm = DecisionTreeClassifier(**params)
    elif algorithm_code == ALGO_CODE_LOGISTIC_REGRESSION:
        algorithm = LogisticRegression(**params)
    elif algorithm_code == ALGO_CODE_GAUSSIAN_PROCESS:
        algorithm = GaussianProcessClassifier(**params)
    elif algorithm_code == ALGO_CODE_RANDOM_FOREST:
        algorithm = RandomForestClassifier(**params)
    elif algorithm_code == ALGO_CODE_EXTRA_TREE:
        algorithm = ExtraTreesClassifier(**params) # n_estimators=250, random_state=0)
    elif algorithm_code == ALGO_CODE_MULTILAYER_PERCEPTRON_CLF:
        algorithm = MLPClassifier(**params)
    elif algorithm_code == ALGO_CODE_MULTILAYER_PERCEPTRON_REG:
        algorithm = MLPRegressor(**params)
    elif algorithm_code == ALGO_CODE_BERNOULLI_RBM:
        algorithm = BernoulliRBM(**params)
    elif algorithm_code == ALGO_CODE_QUADRATIC_DA:
        algorithm = QuadraticDiscriminantAnalysis(**params)
    else:
        raise Exception("invalid algorithm code")

    return algorithm

def get_classifier_algorithm_config(algorithm_code, params=None):

    if params is None:
        params = _ger_algorithm_default_params(algorithm_code=algorithm_code)

    instance = _get_algorithm_instance(algorithm_code, params)

    cc = ClassifierAlgorithmConfig(
        algorithm_code=algorithm_code,
        algorithm_instance=instance,
        algorithm_params=params,
        dense_matrix_required=True if algorithm_code in ALGO_CODE_REQUIRING_DENSE_MATRIX else False,
    )

    return cc
