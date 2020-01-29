"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of URMiner and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""


# on dep. injection http://python-dependency-injector.ets-labs.org/introduction/di_in_python.html
import logging

import dependency_injector.containers as containers
import dependency_injector.providers as providers

from URMiner.classifier_ur.data import data_provider

class URTruthsetHandlerFactory(containers.DeclarativeContainer):
    """IoC container of truth set handlers."""

    ur_sentence = providers.Factory(data_provider.URTruthsetSentenceHandler)
    ur_review = providers.Factory(data_provider.URTruthsetReviewHandler)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    ur_s = URTruthsetHandlerFactory.ur_sentence()
