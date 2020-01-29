"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of the Rationalytics framework and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

from abc import ABC, abstractmethod

class AClassifierOptimizerStrategy(ABC):
    def __init__(self, classifier_cfg):
        self._classifier_cfg = classifier_cfg

    def do_optimization(self):
        """ It should be possible to implement this here (not in sub-classes)
        """
        pass

    @abstractmethod
    def get_high_precision_classifier_config(self):
        pass

    @abstractmethod
    def get_high_recall_classifier_config(self):
        pass

