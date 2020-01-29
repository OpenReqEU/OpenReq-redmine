import logging
from json import loads

from flask import Flask

from services import feature_extractor_bp

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

#
# test data
data_param1 = {"text_value": "I bought this product because I am a long time committed Quicken user, "
                             "and I was not willing to switch to the Mac version when I bought a iMac computer, "
                             "based on the terrible reviews of the software.Parallels has made it possible for me "
                             "to run Quicken 2014 easily.My only complaint is that my son likes to play video games "
                             "and I tried to use parallels with Steam and their game Space Engineers, but the video "
                             "adapter used by Parallels isn't compatible."}

data_param2 = {"text_value":    "This is a shorter text."}

data_param3 = {"text_value":    "Fortunately, was able to reload Quicken 2013 from CD.  " \
                                "Reminder to users who password-protect your data (and who regularly change that password): " \
                                "keep a log of your passwords since it may be difficult to recall what you were using with " \
                                "an older version of the software."}


def test_sentence_count():

    app = Flask(__name__)
    app.testing = True
    bp = feature_extractor_bp.get_blueprint()
    app.register_blueprint(bp)

    # test sentence count
    test_client = app.test_client()
    response = test_client.post(feature_extractor_bp.URI_SENTENCE_COUNT, data=data_param1)
    response_json = loads(response.data)

    # ensure one sentence
    assert response_json[feature_extractor_bp.RESPONSE_DATA_KEY] == 1


def test_extract_postag_clause_level():

    app = Flask(__name__)
    app.testing = True
    bp = feature_extractor_bp.get_blueprint()
    app.register_blueprint(bp)

    #
    # test with data_param1
    test_client = app.test_client()
    response = test_client.post(feature_extractor_bp.URI_POSTAGS_CLAUSES_STRING,
                                             data=data_param1)
    response_json = loads(response.data)
    print (response_json[feature_extractor_bp.RESPONSE_DATA_KEY])

    #
    # test with data_param2
    response = test_client.post(feature_extractor_bp.URI_POSTAGS_CLAUSES_STRING,
                                             data=data_param2)
    response_json = loads(response.data)
    print (response_json[feature_extractor_bp.RESPONSE_DATA_KEY])

def test_extract_postag_phrase_level():

    app = Flask(__name__)
    app.testing = True
    bp = feature_extractor_bp.get_blueprint()
    app.register_blueprint(bp)

    for d in [data_param1,data_param2,data_param3]:

        # test with data
        test_client = app.test_client()
        response = test_client.post(feature_extractor_bp.URI_POSTAGS_PHRASES_STRING, data=d)
        response_json = loads(response.data)
        print (response_json[feature_extractor_bp.RESPONSE_DATA_KEY])
