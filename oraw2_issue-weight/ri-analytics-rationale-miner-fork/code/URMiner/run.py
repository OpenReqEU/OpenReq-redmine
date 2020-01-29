"""
created:    16.05.2018
author:     Volodymyr Biryuk

Module for running the logging server within PyCharm (only for development).
"""
from URMiner.app import urminer_app

if __name__ == '__main__':
    urminer_app.run(host='0.0.0.0', debug=True, port=9704)
