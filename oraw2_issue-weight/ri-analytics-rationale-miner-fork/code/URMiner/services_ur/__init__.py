import logging

from flask_restful.reqparse import RequestParser

# from classifier_ur import URClassifier
from URMiner.classifier_ur.data import COL_TEXT_TITLE, COL_TEXT_BODY, COL_RATING
from URMiner.classifier_ur.ml.classifier_review import URBaselineReviewClassifierConfig, URNewReviewClassifierConfig
#from URMiner.classifier_ur.ml.classifier_sentence import URBaselineSentenceClassifierConfig
from URMiner.classifier_ur.ml.classifier_ur import URClassifier

URI_ROOT_REVIEW_MINER = "/hitec/ur-review-miner"
URI_ROOT_SENTENCE_MINER = "/hitec/ur-sentence-miner"

RESPONSE_DATA_KEY = "data"
RESPONSE_ERROR_KEY = "error"

# def _get_sentence_clf(ur_label, tag):
#     logging.debug("Prepare issue sentence clf")
#     cfg = URBaselineSentenceClassifierConfig()
#     cfg.set_default_classifier_algorithm_via_code("nb")

#     clf = URClassifier(cfg, ur_label, tag)
#     return clf

def _get_review_clf(ur_label, tag):
    logging.debug("Prepare issue review clf")
    cfg = URBaselineReviewClassifierConfig()
    cfg.set_default_classifier_algorithm_via_code("nb")

    clf = URClassifier(cfg, ur_label, tag)
    return clf

def _get_new_review_clf(ur_label, tag):
    logging.debug("Prepare issue review clf")
    cfg = URNewReviewClassifierConfig()
    cfg.set_default_classifier_algorithm_via_code("nb")

    clf = URClassifier(cfg, ur_label, tag)
    return clf



review_request_parser = RequestParser(bundle_errors=True)
review_request_parser.add_argument(COL_TEXT_TITLE, type=str, required=True, help="%s has to be valid string" % COL_TEXT_TITLE)
review_request_parser.add_argument(COL_TEXT_BODY, type=str, required=True, help="%s has to be valid string" % COL_TEXT_BODY)
review_request_parser.add_argument(COL_RATING, type=int, required=True, help="%s needs to be a valid integer" % COL_RATING)

# sentence_request_parser = RequestParser(bundle_errors=True)
# sentence_request_parser.add_argument(COL_TEXT_TITLE, type=str, required=False, help="%s has to be valid string" % COL_TEXT_TITLE)
# sentence_request_parser.add_argument(COL_TEXT_BODY, type=str, required=True, help="%s has to be valid string" % COL_TEXT_BODY)
# sentence_request_parser.add_argument(COL_RATING, type=int, required=True, help="%s needs to be a valid integer" % COL_RATING)