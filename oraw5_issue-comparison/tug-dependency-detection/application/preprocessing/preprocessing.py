# -*- coding: utf-8 -*-

import logging
import os
import csv
from application.entities.requirement import Requirement
from application.util import helper
from functools import reduce
from application.preprocessing import tokenizer
from application.preprocessing import filters
from application.preprocessing import stopwords
from application.preprocessing import stemmer


_logger = logging.getLogger(__name__)


def replace_adjacent_token_synonyms_and_remove_adjacent_stopwords(posts):
    '''
        Looks for adjacent tokens in each post as defined in the synonym list
        and replaces the synonyms according to the synonym list.

        Note: Synonyms that are assigned to no/empty target word in the list are considered
              as 2- or 3-gram stopwords_en and removed.

        The synonym list mainly covers the most frequent 1-gram, 2-gram and 3-grams
        of the whole 'programmers.stackexchange.com' dataset (after our tokenization,
        stopword-removal, ...) as analyzed by using scikitlearn's Count-vectorizer.

        --------------------------------------------------------------------------------------------
        NOTE: Please keep in mind that this method is executed BEFORE stemming, so the list
              may contain slightly different versions of the same synonym words (e.g. plurals, ...)
              This is useful for some context-based words where stemming fails.

              Doing the synonym replacement step before stemming makes the synonym list much more
              readable.
        --------------------------------------------------------------------------------------------
    '''
    synonyms_file_path = os.path.join(helper.APP_PATH, 'corpora', 'tokens', 'synonyms')
    token_replacement_map_unigram = {}
    token_replacement_map_bigram = {}
    token_replacement_map_trigram = {}
    with open(synonyms_file_path, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            source_token_ngram = row[1].strip()
            source_token_parts = source_token_ngram.split()
            target_token_parts = row[0].strip().split()
            if len(source_token_parts) == 1:
                #assert(row[1] not in token_replacement_map_unigram, "Synonym entry '%s' is ambiguous." % row[1])
                token_replacement_map_unigram[source_token_ngram] = target_token_parts
            elif len(source_token_parts) == 2:
                #assert(row[1] not in token_replacement_map_bigram, "Synonym entry '%s' is ambiguous." % row[1])
                token_replacement_map_bigram[source_token_ngram] = target_token_parts
            elif len(source_token_parts) == 3:
                #assert(row[1] not in token_replacement_map_trigram, "Synonym entry '%s' is ambiguous." % row[1])
                token_replacement_map_trigram[source_token_ngram] = target_token_parts
            #else:
            #    assert(False, "Invalid entry in synonyms list! Only supported: unigrams, bigrams, trigrams")

    n_replacements_total = 0
    for post in posts:
        assert(isinstance(post, Post))

        def _replace_token_list_synonyms(tokens, token_replacement_map, n_gram=1):
            assert(isinstance(tokens, list))
            n_replacements = 0
            if len(tokens) < n_gram:
                return (tokens, n_replacements)

            new_tokens = []
            skip_n_tokens = 0
            for i in range(len(tokens)):
                # simplify in order to avoid redundant loop iterations...
                if skip_n_tokens > 0:
                    skip_n_tokens -= 1
                    continue

                if i + n_gram > len(tokens):
                    new_tokens += tokens[i:]
                    break

                n_gram_word = ' '.join(tokens[i:i+n_gram])
                if n_gram_word in token_replacement_map:
                    new_tokens += token_replacement_map[n_gram_word]
                    skip_n_tokens = (n_gram - 1)
                    n_replacements += 1
                else:
                    new_tokens += [tokens[i]]
            return (new_tokens, n_replacements)

        # title tokens
        tokens = post.title_tokens
        tokens, n_replacements_trigram = _replace_token_list_synonyms(tokens, token_replacement_map_trigram, n_gram=3)
        assert(isinstance(tokens, list))
        tokens, n_replacements_bigram = _replace_token_list_synonyms(tokens, token_replacement_map_bigram, n_gram=2)
        tokens, n_replacements_unigram = _replace_token_list_synonyms(tokens, token_replacement_map_unigram, n_gram=1)
        # adjacent stop words have been replaced with empty string! -> remove empty tokens now!
        tokens = filter(lambda t: len(t) > 0, tokens)
        post.title_tokens = list(tokens)
        n_replacements_total += n_replacements_trigram + n_replacements_bigram + n_replacements_unigram

        # body tokens
        tokens = post.body_tokens
        tokens, n_replacements_trigram = _replace_token_list_synonyms(tokens, token_replacement_map_trigram, n_gram=3)
        tokens, n_replacements_bigram = _replace_token_list_synonyms(tokens, token_replacement_map_bigram, n_gram=2)
        tokens, n_replacements_unigram = _replace_token_list_synonyms(tokens, token_replacement_map_unigram, n_gram=1)
        # adjacent stop words have been replaced with empty string! -> remove empty tokens now!
        tokens = filter(lambda t: len(t) > 0, tokens)
        post.body_tokens = list(tokens)
        n_replacements_total += n_replacements_trigram + n_replacements_bigram + n_replacements_unigram

    _logger.info("Found and replaced %s synonym tokens", n_replacements_total)


def _to_lower_case(requirements):
    _logger.info("Lower case requirement title and description")
    for requirement in requirements:
        assert(isinstance(requirement, Requirement))
        requirement.title = requirement.title.lower()
        requirement.description = requirement.description.lower()


def _replace_german_umlauts(requirements):
    _logger.info("Replace umlauts")
    for requirement in requirements:
        assert(isinstance(requirement, Requirement))
        requirement.title = helper.replace_german_umlaut(requirement.title)
        requirement.description = helper.replace_german_umlaut(requirement.description)


def _remove_german_abbreviations(requirements):
    _logger.info("Remove abbreviations")
    for requirement in requirements:
        assert(isinstance(requirement, Requirement))
        requirement.title = requirement.title.replace('z.b.', '')
        requirement.description = requirement.description.replace('z.b.', '')


def _remove_english_abbreviations(requirements):
    _logger.info("Remove abbreviations")
    for requirement in requirements:
        assert(isinstance(requirement, Requirement))
        requirement.title = requirement.title.replace('e.g.', '')
        requirement.title = requirement.title.replace('i.e.', '')
        requirement.title = requirement.title.replace('in order to', '')
        requirement.description = requirement.description.replace('e.g.', '')
        requirement.description = requirement.description.replace('i.e.', '')
        requirement.description = requirement.description.replace('in order to', '')


def preprocess_requirements(requirements, enable_stemming=False, lang="en"):
    _logger.info("Preprocessing requirements")
    assert(isinstance(requirements, list))
    assert(len(requirements) > 0)

    _to_lower_case(requirements)
    if lang == "de":
        _replace_german_umlauts(requirements)
    elif lang == "en":
        _remove_english_abbreviations(requirements)

    all_requirement_titles = list(map(lambda requirement: requirement.title, requirements))
    important_key_words = tokenizer.key_words_for_tokenization(all_requirement_titles)
    _logger.info("Number of key words {} (altogether)".format(len(important_key_words)))
    tokenizer.tokenize_requirements(requirements, important_key_words, lang=lang)
    n_tokens = reduce(lambda x, y: x + y, map(lambda t: len(list(t.title_tokens)) + len(list(t.description_tokens)), requirements))
    filters.filter_tokens(requirements, important_key_words)
    stopwords.remove_stopwords(requirements, lang=lang)

    if enable_stemming is True:
        _logger.warning("Stemming enabled!")
        stemmer.porter_stemmer(requirements)

    n_filtered_tokens = n_tokens - reduce(lambda x, y: x + y, map(lambda t: len(list(t.title_tokens)) + len(list(t.description_tokens)), requirements))
    if n_tokens > 0:
        _logger.info("Removed {} ({}%) of {} tokens (altogether)".format(n_filtered_tokens,
                     round(float(n_filtered_tokens) / n_tokens * 100.0, 2), n_tokens))
    return requirements

