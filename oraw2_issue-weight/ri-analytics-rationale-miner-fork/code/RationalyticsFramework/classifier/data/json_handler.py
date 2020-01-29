"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of the Rationalytics framework and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

import json

def json_dump(data, filename):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile, indent=2)


def json_load(filename):
    with open(filename) as json_data:
        data = json.load(json_data)

    return data