"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of the Rationalytics framework and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

import logging
from collections import Counter

import eli5
import numpy as np

from RationalyticsFramework.classifier.ml.configs.classifier_configs_algorithm import ALGO_CODE_EXTRA_TREE, get_classifier_algorithm_config
from RationalyticsFramework.classifier.ml.trainer.classifier_trainer import make_classifier_training_pipeline


def _get_feature_names(feature_union):

    v = []
    #
    # iterate over all preprocessors and
    # stack horizontally the feature names vectors of each
    for tl in feature_union.transformer_list:
        # fe.transformer_list[0][1]._final_estimator.get_feature_names()
        # 1: accesses the pipeline, _final_estimator: TfIdf...
        steps = tl[1].steps
        for i in  reversed(range(0, len(steps))):
            # search for the first estimator having get_feature_names defined
            # steps[i] returned tuple (name, estimator)
            has_getter = hasattr(steps[i][1], 'get_feature_names')
            if has_getter:
                logging.debug("step name: %s" % steps[i][0])
                logging.debug("%s providing feature names" % steps[i][1].__class__.__name__)
                tmp = ["%s (%s)" % (fname, steps[i][0]) for fname in steps[i][1].get_feature_names()]
                v += tmp
                break
    return v

def assess_best_features(cfg, granularity, label_col, algorithm_code=ALGO_CODE_EXTRA_TREE):

    logging.debug("Assess best features for %s, on %s level, using algorithm_code %s" %
                  (granularity, label_col, algorithm_code))

    data_cfg = cfg.get_data_sample_config(label_col=label_col, max_items_per_class=None)
    # classifier_cfg = cfg.get_classifier_algorithm_config(algorithm_code)

    pipe = make_classifier_training_pipeline(cfg, get_classifier_algorithm_config(algorithm_code))

    #
    # prepare the data
    X = data_cfg.df
    y = data_cfg.df[label_col]

    #
    # fit the pipeline
    pipe.fit(X, y)

    # get the classifier instance
    forest_clf = pipe.named_steps["classifier"].named_steps[algorithm_code]

    # get feature union (that include tranformers)
    fe = pipe.named_steps["data_preprocessor"].named_steps["feature_union"]

    feature_names = _get_feature_names(fe)
    feature_importances = forest_clf.feature_importances_
    indices = np.argsort(feature_importances)[::-1]

    eli5_output = eli5.show_weights(forest_clf, feature_names=feature_names, top=100)
    # eli5_output = eli5.show_prediction(forest_clf, X.iloc[5], feature_names=feature_names, top=10, show_feature_values=True)
    from IPython.core.display import display
    display(eli5_output)

    # compute standard deviation
    std = np.std([tree.feature_importances_ for tree in forest_clf.estimators_],
                 axis=0)

    # # Print the feature ranking
    print("Feature ranking:")
    for i in range(100):
        print("%d. feature %d, %s, (%f, std:%f)" % (i + 1, indices[i], feature_names[indices[i]], feature_importances[indices[i]], std[indices[i]]))

    print ("\nAverage standard deviation: %s" % np.average(std))

    return indices, feature_names, feature_importances, std

def print_best_features_for_latex_table(cfg, granularity, label_cols, fature_count=10):
    final_vars = Counter()

    for l in label_cols:

        indices, feature_names, feature_importances, std = assess_best_features(cfg, granularity, l)

        final_vars[l] = {}
        final_vars[l]["indices"] = indices
        final_vars[l]["feature_names"] = feature_names
        final_vars[l]["feature_importances"] = feature_importances
        final_vars[l]["std"] = std

    latex_rows = ""
    for i in range(fature_count):
        if (i+1) % 2 == 0:
            latex_rows += "\\rowcolor{lightgray2}\n"

        latex_rows += "%i " % (i+1)
        for l in label_cols:
            indices = final_vars[l]["indices"]
            latex_rows += "& %s" % final_vars[l]["feature_names"][indices[i]]
        latex_rows += " \\\\ \n"

    print (latex_rows)

def print_best_features_for_latex_table_single_labelcol(cfg, granularity, label_col, feature_count=10):
    final_vars = Counter()

    indices, feature_names, feature_importances, std = assess_best_features(cfg, granularity, label_col)

    final_vars[label_col] = {}
    final_vars[label_col]["indices"] = indices
    final_vars[label_col]["feature_names"] = feature_names
    final_vars[label_col]["feature_importances"] = feature_importances
    final_vars[label_col]["std"] = std

    latex_rows = ""
    latex_rows_arr = [""] * feature_count

    for i in range(feature_count):
        i_latexrow = i % int(feature_count/2)

        if i <= int(feature_count/2):
            if (i + 1) % 2 == 0:
                latex_rows_arr[i_latexrow] += "\\rowcolor{lightgray2}\n"

            latex_rows_arr[i_latexrow] += "%i " % (i+1)
        else:
            latex_rows_arr[i_latexrow] += "& %i " % (i + 1)

        indices = final_vars[label_col]["indices"]
        latex_rows_arr[i_latexrow] += "& %s" % final_vars[label_col]["feature_names"][indices[i]]

        if i > int(feature_count/2):
            latex_rows_arr[i_latexrow] += " \\\\ \n"

    for i_row in latex_rows_arr:
        print (i_row)