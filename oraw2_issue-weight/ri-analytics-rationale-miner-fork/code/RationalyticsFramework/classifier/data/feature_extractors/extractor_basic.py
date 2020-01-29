"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of the Rationalytics framework and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

# from abc import ABC
#
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.feature_extraction.text import CountVectorizer
#
#
from abc import ABC

from sklearn.feature_extraction.text import CountVectorizer


class AFeatureExtractor(ABC):

    def __init__(self):
        self.cvec = CountVectorizer()


    def fit(self, x, y=None):
        self.cvec.fit(x)

        return self


    def transform(self, data_dict):
        pass