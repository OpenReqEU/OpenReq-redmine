"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of the Rationalytics framework and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

import string
import os

import nltk as nltk
import spacy
import pandas as pd
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
# from spacy.en import English


NLTK_LEMMATIZER = nltk.WordNetLemmatizer()
# SPACY_NLP = spacy.load('en')
SPACY_NLP = spacy.load('fr')

def remove_stopwords(txt_list):
    out = []
    for i, value in enumerate(txt_list):
        value_nostops = ' '
        if not pd.isnull(value):
            value_nostops = remove_stopwords_from_text(value)
            out.append(value_nostops)
    return out

def remove_stopwords_from_text(txt):
    tokens = nltk.word_tokenize(txt.lower())

    tokens = [w for w in tokens if w not in ENGLISH_STOP_WORDS]

    txt_nostops = ' '.join(tokens)

    return txt_nostops

def remove_punctuation(txt_list):
    out = []
    for i, value in enumerate(txt_list):
        value_nopunct = ' '
        if not pd.isnull(value):
            value_nopunct = remove_punctuation_from_text(value)
            out.append(value_nopunct)
    return out

def remove_punctuation_from_text(txt):
    tokens = nltk.word_tokenize(txt.lower())

    tokens = [w for w in tokens if w not in string.punctuation]

    txt_nostops = ' '.join(tokens)

    return txt_nostops

def do_lemmatize(text_list, lemmatizer="SPACY"):

    Value_lemmatized = []

    if lemmatizer == "NLTK":
        # init lemmatizer
        lemmatizer = nltk.WordNetLemmatizer()

        for i, value in enumerate(text_list):
            # print type(value)
            value_lemmatized = ' '
            if not pd.isnull(value):
                value_lemmatized = do_lemmatize_text_nltk(value)

            Value_lemmatized.append(value_lemmatized)

    if lemmatizer == "SPACY":
        for i, value in enumerate(text_list):
            # print("%i" % i)
            value_lemmatized = ' '
            if not pd.isnull(value):
                value_lemmatized = do_lemmatize_text_spacy(value)

            Value_lemmatized.append(value_lemmatized)

    return Value_lemmatized

def do_stem(text_list):
    #todo implement stemmer from nltk
    pass

def do_lemmatize_text_nltk(txt):
    # print type(value)
    txt_lemmatized = ' '
    for word in nltk.word_tokenize(txt):
        txt_lemmatized += NLTK_LEMMATIZER.lemmatize(word.strip()) + ' '

    txt_lemmatized = txt_lemmatized[:-1]

    return txt_lemmatized

def do_lemmatize_text_spacy(txt):
    # print type(value)
    txt_lemmatized = ' '
    txt_processed = SPACY_NLP(txt)

    for token in txt_processed:
        txt_lemmatized += token.lemma_ + " "

    txt_lemmatized = txt_lemmatized[:-1]

    return txt_lemmatized