"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of URMiner and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

from URMiner.classifier_ur.data import UR_SENTENCE_CSV_FILE, UR_REVIEW_CSV_FILE
from RationalyticsFramework.classifier.data import GRANULARITY_SENTENCE, GRANULARITY_COMMENT
from RationalyticsFramework.classifier.data.data_provider import ASingleGranularityTruthsetHandler

class URTruthsetSentenceHandler(ASingleGranularityTruthsetHandler):
    def __init__(self):
        super().__init__(granularity=GRANULARITY_SENTENCE,
                         source_file=UR_SENTENCE_CSV_FILE,
                         class_labels=[True, False])

class URTruthsetReviewHandler(ASingleGranularityTruthsetHandler):

    def __init__(self):
        super().__init__(granularity=GRANULARITY_COMMENT,
                         source_file=UR_REVIEW_CSV_FILE,
                         class_labels=[True, False])
