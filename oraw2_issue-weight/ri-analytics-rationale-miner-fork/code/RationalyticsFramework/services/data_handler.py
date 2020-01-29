"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of the Rationalytics framework and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

from pymongo import MongoClient

db_client = MongoClient('mongodb://localhost:27017/asr')

db_asr = db_client['asr']
asr_review_coll = db_asr['asr_review_new']
asr_sentence_coll = db_asr['asr_sentence_new']

def get_asr_sentence(page_number, per_page):
    page_number = int(page_number)
    per_page = int(per_page)

    return asr_sentence_coll.find().skip((page_number-1)*per_page).limit(per_page);

def get_asr_review(page_number, per_page):
    page_number = int(page_number)
    per_page = int(per_page)

    return asr_review_coll.find().skip((page_number - 1) * per_page).limit(per_page);


DEBUG = False
if DEBUG:

    for o in get_asr_sentence(1,20):
        print (o['Value'])