"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of the Rationalytics framework and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

import nltk
import spacy
import os

from collections import deque

# SPACY_NLP = spacy.load('en')
SPACY_NLP = spacy.load('fr')

def do_extract_sentences(text):

    nlp_result = SPACY_NLP(text)
    sentences = [sent.string.strip() for sent in nlp_result.sents]

    return sentences


def extract_sentence_count(text_list):

    text_list_counts = []
    for text in text_list:
        text_list_counts.append(extract_sentence_count_from_text(text))

    return text_list_counts


def extract_sentence_count_from_text(text):

    sent_list = nltk.sent_tokenize(text)

    sent_list_stripped = [sent.strip() for sent in sent_list]

    return len(sent_list_stripped)


def extract_word_count(text_list):

    text_list_counts = []
    for text in text_list:
        text_list_counts.append(extract_word_count_from_text(text))

    return text_list_counts


def extract_word_tokens(txt, normalize=True):
    word_list = nltk.word_tokenize(txt)

    if normalize:
        word_list = [sent.lower().strip() for sent in word_list]

    return word_list

def extract_word_count_from_text(txt):

    word_list = nltk.word_tokenize(txt)

    word_list_stripped = [sent.strip() for sent in word_list]

    return len(word_list_stripped)

def get_first_sentence(txt):
    text_sentences = SPACY_NLP(txt).sents

    first_sentence = next(text_sentences)

    return first_sentence

def get_last_sentence(txt):
    text_sentences = SPACY_NLP(txt).sents

    dd = deque(text_sentences, maxlen=1)
    last_sentence = dd.pop()

    return last_sentence




