"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of the Rationalytics framework and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

import logging
from abc import abstractmethod, ABC

from RationalyticsFramework.classifier.ml.trainer.classifier_trainer import check_model_file_exists, train_and_persist_classifier_model, \
    load_classifier_model


class ARationalyticsClassifier(ABC):

    def __init__(self, model_folder, classifier_cfg, tag, label_column): #allow plural?

        self._model_folder = model_folder
        self._label_column = label_column
        self._classifier_cfg = classifier_cfg
        self._classifier_algo_cfg = classifier_cfg.get_default_classifier_algorithm()
        self._base_filename = "%s%s_%s_%s" % (self._model_folder, self._label_column, self._classifier_algo_cfg.algorithm_code, tag)
        self._model = None

    def train(self, use_cache=True):

        logging.debug("Train classifier...")
        if use_cache:
            file_exists = check_model_file_exists(self._base_filename)
            if file_exists:
                logging.debug("Load model from file..")
                self._model = load_classifier_model(self._base_filename)
                # done!
                return

        # train & persist classification model
        self._model = train_and_persist_classifier_model(classifier_cfg=self._classifier_cfg,
                                           label_column=self._label_column,
                                           base_filename=self._base_filename)

    @abstractmethod
    def predict(self, **kwargs):
        pass

    @abstractmethod
    def predict_list(self, **kwargs):
        pass