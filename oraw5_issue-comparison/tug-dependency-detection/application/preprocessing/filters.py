# -*- coding: utf-8 -*-

import logging
import re
import os
from application.util import helper


_logger = logging.getLogger(__name__)
emoticons_data_file = os.path.join(helper.APP_PATH, "corpora", "emoticons", "emoticons")

tokens_punctuation_re = re.compile(r"(\.|!|\?|\(|\)|~)$")

KNOWN_FILE_EXTENSIONS_MAP = {
    "exe": "windows",
    "jar": "java",
    "js": "javascript",
    "h": "c",
    "py": "python",
    "s": "assembly",
    "rb": "ruby"
}


def filter_tokens(requirements, important_key_words):
    _logger.info("Filter posts' tokens")
    regex_url = re.compile(
        r'^(?:http|ftp)s?://'  # http:// https:// ftp:// ftps://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    # from nltk.tokenize.casual
    regex_emoticons = re.compile(r"""
        (?:
          [<>]?
          [:;=8]                     # eyes
          [\-o\*\']?                 # optional nose
          [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
          |
          [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
          [\-o\*\']?                 # optional nose
          [:;=8]                     # eyes
          [<>]?
          |
          <3                         # heart
        )""", re.IGNORECASE)

    regex_hex_numbers = re.compile(r'^0?x[0-9a-fA-F]+$', re.IGNORECASE)
    regex_number = re.compile(r'^#\d+$', re.IGNORECASE)
    #regex_float_number = re.compile(r'^\d+\.\d+$', re.IGNORECASE)
    regex_color_code = re.compile(r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$', re.IGNORECASE)
    #regex_long_number_in_separated_format = re.compile(r'^\d+,\d+(,\d+)?$', re.IGNORECASE)

    with open(emoticons_data_file) as emoticons_file:
        emoticons_list = emoticons_file.read().splitlines()

    def _filter_tokens(tokens, important_key_words):
        # remove urls
        tokens = [word for word in tokens if regex_url.match(word) is None]

        # also remove www-links that do not start with "http://" or "https://"!!
        tokens = filter(lambda t: not t.startswith("www."), tokens)

        # remove emoticons (list from https://en.wikipedia.org/wiki/List_of_emoticons)
        tokens = [word for word in tokens if word not in emoticons_list]

        # remove more-complex emoticons (regex)
        tokens = [word for word in tokens if regex_emoticons.match(word) is None]

        # remove words that start or end with "_"
        tokens = filter(lambda t: not t.startswith("_") and not t.endswith("_"), tokens)

        new_tokens = []
        for t in tokens:
            if helper.is_int_or_float(t) or helper.is_int_or_float(t.replace(',', '')):
                new_tokens.append([t])
                continue

            # allow tag names
            if t in important_key_words:
                new_tokens.append([t])
                continue

            # allow numbers
            if t.replace(".", "", 1).replace(",", "", 1).isdigit():
                new_tokens.append([t])
                continue

            separator = None
            for sep in ["-", ".", "_", ",", "/"]:
                if sep in t:
                    separator = sep
                    break
            if separator is None:
                new_tokens.append([t])
                continue

            # split single word by separator and treat each part as a single token!
            parts = t.split(separator)
            assert len(parts) > 1
            if len(parts) == 2 and separator == ".":
                if parts[1] in KNOWN_FILE_EXTENSIONS_MAP:
                    new_tokens.append(KNOWN_FILE_EXTENSIONS_MAP[parts[1]])
                    continue
            new_tokens.append(parts)
        tokens = [t for sub_tokens in new_tokens for t in sub_tokens]

        # remove empty tokens
        tokens = filter(lambda t: len(t) > 0, tokens)

        # remove single- and dual-character words that are not part of our the important_key_words list
#         tokens = filter(lambda t: len(t) > 2 or t in important_key_words, tokens)

        #-------------------------------------------------------------------------------------------
        # Note: We figured out not removing numbers slightly increases the performance of our models
        #       especially when using bigrams or trigrams:
        #       -> e.g. "windows", "2008" -> "windows 2008"
        #           or: "web", "2.0" -> "web 2.0"
        #
        # remove . and , separated numbers and enumerations!
        #tokens = filter(lambda t: regex_float_number.match(t) is None, tokens)
        #tokens = filter(lambda t: regex_long_number_in_separated_format.match(t) is None, tokens)
        #-------------------------------------------------------------------------------------------

#         # remove twitter-like @-mentions (e.g. @peter, @all)
#         tokens = filter(lambda t: not t.startswith("@"), tokens)

        # make sure that all tokens do not contain any whitespaces before and at the end
        tokens = map(lambda t: t.strip(), tokens)

        # remove empty tokens
        return list(filter(lambda t: len(t) > 0, tokens))

    for requirement in requirements:
        requirement.title_tokens = _filter_tokens(requirement.title_tokens, important_key_words)
        requirement.description_tokens = _filter_tokens(requirement.description_tokens, important_key_words)

