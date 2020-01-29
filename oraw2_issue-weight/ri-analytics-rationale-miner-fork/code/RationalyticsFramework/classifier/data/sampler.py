"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of the Rationalytics framework and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

from abc import abstractmethod, ABC

class ASampler(ABC):
    @abstractmethod
    def get_balanced_sample_data_cfg(self):
        pass