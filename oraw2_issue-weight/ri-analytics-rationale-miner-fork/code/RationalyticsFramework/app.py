"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of the Rationalytics framework and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

import jinja2 as j2
from flask import Flask, render_template

from services import preprocessor_bp, data_handler_bp, feature_extractor_bp

#
# configure logging
import logging
logging.basicConfig(level=logging.DEBUG)

#
# connect to the MongoDB
# connect(host='mongodb://localhost:27017/asr')

#
# define jinja2 template environment
template_env = j2.Environment(
        loader=j2.PackageLoader('app','templates'),
        autoescape=j2.select_autoescape(['html', 'xml', 'tpl']),
        enable_async= True,
        trim_blocks = True,
        lstrip_blocks = True
)

app = Flask(__name__, static_folder='assets')
# app.static_folder('/static', './static')
# app.static_folder('/assets', './assets')
# app.static_folder('/_playground', './_playground')

# for debug only
# app.static_folder('/_playground/html', './_playground/html')

@app.route('/assets/<file>')
def static_proxy(file):
  # send_static_file will guess the correct MIME type
  return app.send_static_file(file)

@app.route("/")
def template_test(request):

    t = template_env.get_template('index.j2.html')

    rendered_template = t.render(
        my_string="Wheeeee!", my_list=[0, 1, 2, 3, 4, 5]
    )

    return render_template(rendered_template)

#
# configure API
app.config['API_VERSION'] = '1.0.0'
app.config['API_TITLE'] = 'Rationalytics API'
app.config['API_DESCRIPTION'] = 'Rationalytics API'
app.config['API_PRODUCES_CONTENT_TYPES'] = ['application/json']
app.config['API_CONTACT_EMAIL'] = 'kurtanovic@informatik.uni-hamburg.de'
app.config['TEMPLATE_ENV'] = template_env

#
# Add all blueprints to this project.
app.register_blueprint(preprocessor_bp.get_blueprint())
app.register_blueprint(data_handler_bp.get_blueprint())
app.register_blueprint(feature_extractor_bp.get_blueprint())

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8085)