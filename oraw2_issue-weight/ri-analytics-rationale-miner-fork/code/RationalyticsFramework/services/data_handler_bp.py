"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of the Rationalytics framework and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

from flask import request, Blueprint, jsonify
from services.data_handler import get_asr_review, get_asr_sentence

_bp = Blueprint("data-handler", __name__)

def get_blueprint():
    return _bp

@_bp.route("/data-handler/<data_source>", methods=["GET"])
def data_handler(data_source):

    if (data_source == "asr-sentence"):
        out = get_asr_sentence(request.args['page'], request.args['per_page'])
    elif (data_source == "asr-review"):
        out = get_asr_review(request.args['page'], request.args['per_page'])

    row_list = []
    for item in out:
        x = {k: item[k] for k in {"CustomReviewId", "RelativeOrder", "ApplicationName", "ValueType", "Value"}}
        row_list.append(x)

    row_list_total = out.count()

    to_return = {"rows" : row_list,
                 "total" : row_list_total}

    return jsonify(to_return)