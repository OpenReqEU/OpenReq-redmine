"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of the Rationalytics framework and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

""" TODO module description
Idea:
- Initialize a AClassifierSelectorStrategy
- run

"""
from abc import ABC, abstractmethod

class AClassifierSelectorStrategy(ABC):
    """
    This class should provide appropriate methods to select
    a classifier model based on the request data payload.
    E.g., if rating is send, and there is BETTER classifier config with a rating-trained model,
    then this model should be selected.
    """
    def __init__(self, request_data):
        self._request_data = request_data

    @abstractmethod
    def get_high_precision_classifier_config(self):
        pass

    @abstractmethod
    def get_high_recall_classifier_config(self):
        pass

    @abstractmethod
    def get_high_F1_classifier_config(self):
        pass


