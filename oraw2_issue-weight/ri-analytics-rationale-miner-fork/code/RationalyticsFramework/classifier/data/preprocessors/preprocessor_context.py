"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of the Rationalytics framework and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

from abc import ABC

from RationalyticsFramework.classifier.data.data_selectors.data_selector_basic import DataPreSelector


class ADataSelectorWithContextSelectStrategy(ABC, DataPreSelector):
    CTOR_PARAM_SELECTOR_STRATEGY_PROVIDER = "selector_strategy_provider"

    def __init__(self, keys, part_selector_strategy_provider):
        super().__init__(keys)
        self.selector_strategy_provider = part_selector_strategy_provider