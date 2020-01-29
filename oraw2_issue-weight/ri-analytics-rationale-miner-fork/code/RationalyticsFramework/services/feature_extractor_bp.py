"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of the Rationalytics framework and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

from flask import Blueprint, jsonify, request
from flask import current_app as app

from base.preprocessor_syntags import extract_postags_string_from_text, extract_wordpos_tuples_from_text, \
    extract_postag_rel_frequency, extract_postags_clause_level_from_text, \
    extract_postags_phrase_level_from_text
from base.preprocessor_basic import extract_sentence_count_from_text, extract_word_count_from_text

_bp = Blueprint("feature-extractor", __name__)

#
# URI and other consts
URI_MAIN = "/feature-extractor"
URI_WORDPOS_TUPLES = URI_MAIN + "/word-pos-tuple"
URI_POSTAGS_STRING = URI_MAIN + "/pos-tags-string"
URI_POSTAG_FREQ = URI_MAIN + "/pos-tag-frequency"
URI_SENTENCE_COUNT = URI_MAIN + "/sentence-count"
URI_WORD_COUNT = URI_MAIN + "/word-count"
URI_POSTAGS_CLAUSES_STRING = URI_MAIN + "/pos-tags-clauses-string"
URI_POSTAGS_PHRASES_STRING = URI_MAIN + "/pos-tags-phrases-string"

RESPONSE_DATA_KEY = "text_feature_extracted"

def get_blueprint():
    return _bp

@_bp.route(URI_MAIN, methods=["GET"])
def get_main():
    '''
    Returns the html test page for this service
    :param request: The request
    :return:
    '''

    t_env = app.config["TEMPLATE_ENV"]
    t = t_env.get_template('tester_feature_extractor.j2.html')

    rendered_template = t.render(
        URI_MAIN = URI_MAIN,
        URI_WORDPOS_TUPLES = URI_WORDPOS_TUPLES,
        URI_POSTAGS_STRING = URI_POSTAGS_STRING,
        URI_POSTAGS_PHRASES_STRING = URI_POSTAGS_PHRASES_STRING,
        URI_POSTAGS_CLAUSES_STRING = URI_POSTAGS_PHRASES_STRING,
        URI_POSTAG_FREQ = URI_POSTAG_FREQ,
        URI_SENTENCE_COUNT = URI_SENTENCE_COUNT,
        RESPONSE_DATA_KEY = RESPONSE_DATA_KEY
    )

    return rendered_template


@_bp.route(URI_WORDPOS_TUPLES, methods=["POST"])
def get_word_pos_tuples():

    p_text = request.form.get("text_value")
    r = {RESPONSE_DATA_KEY: extract_wordpos_tuples_from_text(p_text)}

    return jsonify(r)


@_bp.route(URI_POSTAGS_STRING, methods=["POST"])
def get_postags_string():

    p_text = request.form.get("text_value")
    r = {RESPONSE_DATA_KEY: extract_postags_string_from_text(p_text)}

    return jsonify(r)

@_bp.route(URI_POSTAGS_PHRASES_STRING, methods=["POST"])
def extract_postags_phrases():

    p_text = request.form.get("text_value")
    r = {RESPONSE_DATA_KEY: extract_postags_phrase_level_from_text(p_text)}

    return jsonify(r)

@_bp.route(URI_POSTAGS_CLAUSES_STRING, methods=["POST"])
def extract_postags_clauses():

    p_text = request.form.get("text_value")
    r = {RESPONSE_DATA_KEY: extract_postags_clause_level_from_text(p_text)}

    return jsonify(r)

@_bp.route(URI_POSTAG_FREQ, methods=["POST"])
def get_postag_frequency():

    p_text = request.form.get("text_value")
    p_postag = request.form.get("postag_key")

    r = {RESPONSE_DATA_KEY: extract_postag_rel_frequency(p_text, p_postag)}

    return jsonify(r)

@_bp.route(URI_SENTENCE_COUNT, methods=["POST"])
def get_sentence_count():

    p_text = request.form.get("text_value")

    r = {RESPONSE_DATA_KEY: extract_sentence_count_from_text(p_text)}

    return jsonify(r)

@_bp.route(URI_WORD_COUNT, methods=["POST"])
def get_word_count():

    p_text = request.form.get("text_value")

    r = {RESPONSE_DATA_KEY: extract_word_count_from_text(p_text)}

    return jsonify(r)



