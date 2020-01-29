import logging
from json import loads

from flask import Flask

from services import preprocessor_bp

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

data_param2 = {"text_value" : "This is a shorter text."}

def test_remove_stops():

    app = Flask(__name__)
    app.testing = True
    app.register_blueprint(preprocessor_bp.get_blueprint())

    test_client = app.test_client()

    # test removal of stopwords
    response = test_client.post(preprocessor_bp.URI_REMOVESTOPS, data=data_param1)
    response_json = loads(response.data)
    assert response_json["text_preprocessed"] == "bought product long time committed quicken user , willing switch mac " \
                                                 "version bought imac computer , " \
                                                 "based terrible reviews software.parallels possible run quicken 2014 easily.my " \
                                                 "complaint son likes play video games tried use parallels steam game space engineers , " \
                                                 "video adapter used parallels n't compatible ."
    print(response_json)

    # test removal of stopwords
    response = test_client.post(preprocessor_bp.URI_REMOVESTOPS,
                                            data=data_param2)
    response_json = loads(response.data)
    assert response_json["text_preprocessed"] == 'shorter text .'
    print(response_json)

def test_do_lemmatize():

    app = Flask(__name__)
    app.testing = True
    app.register_blueprint(preprocessor_bp.get_blueprint())

    test_client = app.test_client()

    # test lemmatization
    response = test_client.post(preprocessor_bp.URI_DOLEMMATIZE, data=data_param2)
    response_json = loads(response.data)

    print(response_json)


