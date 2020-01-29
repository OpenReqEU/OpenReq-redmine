"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of URMiner and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

import sys
from flask_cors import CORS

r_framework_path = "../RationalyticsFramework/"
sys.path.append(r_framework_path)


import logging

from flask_restful_swagger_2 import Api

logging.basicConfig(level=logging.ERROR)

from flask import Flask
from URMiner.services_ur import weighting_service#issue_service, alternative_service, decision_service, justification_service, criteria_service


urminer_app = Flask(__name__)

CORS(urminer_app)
urminer_app_api = Api(urminer_app)


# configure API
urminer_app.config['API_VERSION'] = '1.0.0'
urminer_app.config['API_TITLE'] = 'URMiner API'
urminer_app.config['API_DESCRIPTION'] = 'URMiner API'
urminer_app.config['API_PRODUCES_CONTENT_TYPES'] = ['application/json']
urminer_app.config['API_CONTACT_EMAIL'] = 'kurtanovic@informatik.uni-hamburg.de'



weighting_service.add_to_api(urminer_app_api)
# issue_service.add_to_api(urminer_app_api)
# alternative_service.add_to_api(urminer_app_api)
# criteria_service.add_to_api(urminer_app_api)
# decision_service.add_to_api(urminer_app_api)
# justification_service.add_to_api(urminer_app_api)



if __name__ == "__main__":
    print()
    # urminer_app.run(host='0.0.0.0', port=8085, debug=True)
