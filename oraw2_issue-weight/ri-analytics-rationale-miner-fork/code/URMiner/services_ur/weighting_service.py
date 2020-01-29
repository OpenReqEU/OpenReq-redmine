"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of URMiner and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

import logging
import time

from flask import Blueprint, jsonify

from flask_restful import Resource
from flask_restful_swagger_2 import Api
from URMiner.classifier_ur.data import COL_IS_ANOMALY, COL_IS_LOW_PRIORITY, COL_IS_AVG_PRIORITY, COL_IS_HIGH_PRIORITY
from URMiner.services_ur import RESPONSE_DATA_KEY, RESPONSE_ERROR_KEY, _get_new_review_clf, review_request_parser#, _get_review_clf, sentence_request_parser, _get_sentence_clf

URI_REVIEW_ISSUE_CLF = "/issue_weighting"

_bp = Blueprint("issue", __name__)
_api = Api(_bp)

def get_blueprint():
    return _bp

def add_to_api(api):
    api.add_resource(IssueWeight, URI_REVIEW_ISSUE_CLF)

class IssueWeight(Resource):
    issue_anomaly_clf = _get_new_review_clf(COL_IS_ANOMALY, "new_fr")
    issue_low_p_clf = _get_new_review_clf(COL_IS_LOW_PRIORITY, "new_fr")
    issue_avg_p_clf = _get_new_review_clf(COL_IS_AVG_PRIORITY, "new_fr")
    issue_high_p_clf = _get_new_review_clf(COL_IS_HIGH_PRIORITY, "new_fr")

    def post(self):
        req_data = review_request_parser.parse_args()

        #try:
        isAnomalyRatio = self.issue_anomaly_clf.predict_dict(req_data)
        isLowPrioRatio = self.issue_low_p_clf.predict_dict(req_data)
        isAvgPrioRatio = self.issue_avg_p_clf.predict_dict(req_data)
        isHighPrioRatio = self.issue_high_p_clf.predict_dict(req_data)
        timestamp = time.time()

        weight = 0
        weight += isAnomalyRatio['True']
        weight -= isLowPrioRatio['True']
        weight += isHighPrioRatio['True']

        result = {
            'isAnomalyRatio': isAnomalyRatio['True'],
            'isLowPrioRatio': isLowPrioRatio['True'],
            'isAvgPrioRatio': isAvgPrioRatio['True'],
            'isHighPrioRatio': isHighPrioRatio['True'],
            'weight': weight
        }
        result = jsonify({RESPONSE_DATA_KEY: result})
        return result

        # except Exception as e:
        #    print(e)
        #    r = jsonify({RESPONSE_ERROR_KEY: str(e)})
        #    r.status_code = 500

        # return r