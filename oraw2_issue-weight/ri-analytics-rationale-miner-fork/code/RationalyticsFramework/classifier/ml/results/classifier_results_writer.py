"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of the Rationalytics framework and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

import datetime
import logging
import subprocess

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from RationalyticsFramework.classifier.ml.configs.classifier_configs_algorithm import get_all_algo_codes, get_algo_name

CSV_SEPARATOR = ","

def get_git_hash():
    git_hash = subprocess.check_output(["git", "describe", "--always"])
    git_hash = str(git_hash, "UTF-8").strip()

    return git_hash

def prf1_2csv(y_true, y_pred, data_cfg, classifier_cfg, classifier_algo_code, tag, file_suffix=""):

    feature_cfg = classifier_cfg.get_activated_features_cfg_dict()
    classifier_algo_cfg = classifier_cfg.get_classifier_algorithm_config(classifier_algo_code)

    # By default, all labels in ``y_true`` and ``y_pred`` are used in sorted order.
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=data_cfg.classes, average=None)
    #
    # p_mean = np.average(p, weights=s),
    # r_mean = np.average(r, weights=s),
    # f1_mean = np.average(f1, weights=s),
    # s_mean = np.sum(s),

    git_hash = get_git_hash()
    dt = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")

    items = [("tag", [tag] * len(data_cfg.classes)),
             ("label_column", [data_cfg.label_column] * len(data_cfg.classes)),
             ("label_class", data_cfg.classes),
             ("precision", p),
             ("recall", r),
             ("f1", f1),
             ("support", s),
             ("git_hash", [git_hash] * len(data_cfg.classes)),
             ("date_time", [dt] * len(data_cfg.classes)),
             ("classifier_cfg", [classifier_algo_cfg]  * len(data_cfg.classes)),
             ("feature_cfg", [feature_cfg] * len(data_cfg.classes))]

    df = pd.DataFrame.from_items(items)
    return df


def y_array_write2csv(y_array, data_cfg, tag, classifier_algo_code=None, file_suffix=""):

    git_hash = get_git_hash()
    dt = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")

    y_array_str = " ".join(str(x) for x in y_array)
    dict_items = {"tag": tag,
             "algorithm_code": classifier_algo_code if classifier_algo_code else " ",
             "label_column": data_cfg.label_column,
             "git_hash": git_hash,
             "date_time": dt,
             "y_array": y_array_str}

    df = pd.DataFrame(dict_items, index=[0])

    return df
