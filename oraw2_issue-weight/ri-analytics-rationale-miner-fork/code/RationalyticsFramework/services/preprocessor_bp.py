"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of the Rationalytics framework and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

from flask import Blueprint, jsonify, request
from flask import current_app as app

from base.preprocessor import remove_stopwords_from_text, remove_punctuation_from_text, do_lemmatize_text_spacy

_bp = Blueprint("preprocessor", __name__)

# label consts
URI_MAIN = "/preprocessor"
URI_REMOVESTOPS = URI_MAIN + "/remove_stops"
URI_REMOVEPUNCT = URI_MAIN + "/remove_puncts"
URI_DOLEMMATIZE = "/lemmatize"
RESPONSE_DATA_KEY = "text_preprocessed"

dummy_text_list = [
    "I bought this product because I am a long time committed Quicken user, and I was not willing to switch to the Mac version when I bought a iMac computer, based on the terrible reviews of the software.Parallels has made it possible for me to run Quicken 2014 easily.My only complaint is that my son likes to play video games and I tried to use parallels with Steam and their game Space Engineers, but the video adapter used by Parallels isn't compatible.",
    "Lighroom does give more detail control but I find moving between photoshop and scaling for emailed photos a chore that is really simple with Apple.Pity then that Apple is now run by an accountant and profit comes before the customer what company annouces its killing two photography applications months before anyone understands Photos!",
    "If you are into landscape or nature photography or many other areas, the book provides no such tips or tutorials for the image processing you are likely to be doing.Another aspect missing is the why of a given procedure in LR or Photoshop."
    "Version 5 provides us with a more incremental  set of features, not that those are anything to sneeze at.The new healing brush allows us to actually brush away features rather than providing spot healing.",
]

def get_blueprint():
    return _bp

@_bp.route(URI_MAIN, methods=["GET"])
def get_main():

    t_env = app.config["TEMPLATE_ENV"]
    t = t_env.get_template('tester_preprocessor.j2.html')

    rendered_template = t.render(
        URI_MAIN = URI_MAIN,
        URI_REMOVESTOPS = URI_REMOVESTOPS,
        URI_REMOVEPUNCT = URI_REMOVEPUNCT,
        RESPONSE_DATA_KEY = RESPONSE_DATA_KEY
    )

    return rendered_template

@_bp.route(URI_REMOVESTOPS, methods=["POST"])
def remove_stops():

    p_text = request.form.get("text_value")
    r = {RESPONSE_DATA_KEY : remove_stopwords_from_text(p_text)}

    return jsonify(r)

@_bp.route(URI_REMOVEPUNCT, methods=["POST"])
def remove_punct():

    p_text = request.form.get("text_value")

    r = {RESPONSE_DATA_KEY : remove_punctuation_from_text(p_text)}

    return jsonify(r)

@_bp.route(URI_DOLEMMATIZE, methods=["POST"])
def do_lemmatize():

    p_text = request.form.get("text_value")

    r = {RESPONSE_DATA_KEY : do_lemmatize_text_spacy(p_text)}

    return jsonify(r)