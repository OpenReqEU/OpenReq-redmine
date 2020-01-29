"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of the Rationalytics framework and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

import logging
import sys

import nltk
from nltk.parse.stanford import StanfordParser
from numpy import mean

from base.preprocessor_basic import do_extract_sentences, SPACY_NLP

PENN_TAGSET_CLAUSE = ["S",
                    "SBAR",
                    "SBARQ",
                    "SINV",
                    "SQ"]

PENN_TAGSET_PHRASE = [
    "ADJP",
    "ADVP",
    "CONJP",
    "FRAG",
    "INTJ",
    "LST",
    "NAC",
    "NP",
    "NX",
    "PP",
    "PRN",
    "PRT",
    "QP",
    "RRC",
    "UCP",
    "VP",
    "WHADJP",
    "WHAVP",
    "WHNP",
    "WHPP",
    "X",
]

PENN_TAGSET_CLAUSEPHRASE = PENN_TAGSET_CLAUSE + PENN_TAGSET_PHRASE


def do_extract_postags_word_level(text_list):
    text_postags_list = []
    text_postags_txt_list = []

    print ("extract postags...")
    for text in text_list:
        #
        # generate POS-Tags
        tokens = nltk.word_tokenize(text)

        # print tokens
        tokens_postagged = nltk.pos_tag(tokens)
        text_postags_txt_list.append(" ".join([pos for (word, pos) in tokens_postagged]))
        text_postags_list.append(tokens_postagged)

    return text_postags_list, text_postags_txt_list

def extract_postags_clause_level_from_text(txt, use_safe_method = True):
    extract_postags_clause_level([txt], use_safe_method)

def extract_postags_phrase_level_from_text(txt, use_safe_method = True):
    extract_postags_phrase_level([txt], use_safe_method)

def extract_postags_clause_level(txt_list, use_safe_method = True):
    if use_safe_method:
        return _extract_syntags_features_safe(txt_list, [PENN_TAGSET_CLAUSE])
    else:
        return _extract_syntags_features(txt_list, [PENN_TAGSET_CLAUSE])

def extract_postags_phrase_level(txt_list, use_safe_method = True):
    if use_safe_method:
        return _extract_syntags_features_safe(txt_list, [PENN_TAGSET_PHRASE])
    else:
        return _extract_syntags_features(txt_list, [PENN_TAGSET_PHRASE])

def extract_postags_clausephrase_level(txt_list, use_safe_method = True):
    if use_safe_method:
        return _extract_syntags_features_safe(txt_list, [PENN_TAGSET_CLAUSEPHRASE])
    else:
        return _extract_syntags_features(txt_list, [PENN_TAGSET_CLAUSEPHRASE])


# @deprecated
# def extract_postags_old_syntags(txt_list, use_safe_method = True):
#
#     if use_safe_method:
#         x,y,z = _extract_syntags_features_safe(txt_list, [SYNTACTIC_TAGSET])
#     else:
#         x, y, z = _extract_syntags_features(txt_list, [SYNTACTIC_TAGSET])
#
#     return x[0], y, z

def _extract_syntags_features_safe(text_list, tagsets):
    """
    This is a more robust extractor that uses the StanfordParser on the sentence level and then summarizes the output
    :param text_list:
    :param tagsets:
    :return:
    """
    logging.debug("Extract syntags safe mode ...")
    # https://stackoverflow.com/questions/34398804/how-to-parse-large-data-with-nltk-stanford-pos-tagger-in-python
    nltk.internals.config_java(options='-xmx2G')
    sp = StanfordParser(verbose=True)

    syntag_list_final = [[] for i in tagsets]
    tree_height_list_final = []
    subtree_count_list_final = []

    for text in text_list:

        sentences = do_extract_sentences(text)

        syntag_list, tree_height_list, subtree_count_list = _extract_syntags_features(sentences, tagsets, use_stanford_tagger=True, stanford_parser_instance=sp)

        # merge all syntags per tagset
        for i, tagset in enumerate(tagsets):
            tmp_syntag_list = ""
            for s in syntag_list[i]:
                tmp_syntag_list += s

            syntag_list_final[i].append(tmp_syntag_list.strip())

        # merge tree height
        tree_height_list_final.append(round(mean(tree_height_list),2))

        # merge subtree count
        subtree_count_list_final.append(round(mean(subtree_count_list), 2))

    return syntag_list_final, tree_height_list_final, subtree_count_list_final


def _extract_syntags_features(text_list, tagsets, use_stanford_tagger=False, stanford_parser_instance=None):
    """

    :param text_list:
    :param tagsets:
    :param use_stanford_tagger: POS-tagging is done by StanfordParser instead of NLTK
    :param stanford_parser_instance:
    :return:
    """

    if not use_stanford_tagger:
        text_4iteration_list, text_postags_txt_list = do_extract_postags_word_level(text_list)
    else:
        nltk.internals.config_java(options='-xmx4G')
        text_4iteration_list = text_list

    sp = StanfordParser(verbose=True, java_options="-xmx4G") if stanford_parser_instance is None else stanford_parser_instance #path_to_jar="/Users/zkey/tools/stanford-parser/stanford-parser.jar"

    treerep_of_sentences = []
    logging.debug("extract treerep of sentences...")
    for i, st in enumerate(text_4iteration_list):
        try:
            logging.debug ("sentence %s" % i)

            if use_stanford_tagger:
                tmp = sp.raw_parse(st)
            else:
                tmp = sp.tagged_parse(st)

            treerep_of_sentences.append(tmp)
        except:
            logging.error(sys.exc_info())
            logging.error("sentence: ", st)
            treerep_of_sentences.append(None)

        # # return unchanged featuresets
        # return featuresets

    # prepare a list for each tagset
    syntags_lists = ["" for i in tagsets]

    syntactic_tree_heigt_list = []
    syntactic_subtree_count_list = []

    logging.debug("sentence list len: %s " % len(treerep_of_sentences))
    logging.debug("tree list len: %s " % len(treerep_of_sentences))

    for j, tree in enumerate(treerep_of_sentences):
        logging.debug("Tree %s" % j)

        if tree:
            s = next(tree)
            s = nltk.ParentedTree.convert(s)

            for tagset_i, tagset in enumerate(tagsets):

                sentence_tags = []
                # for i, st in enumerate(s.subtrees(filter=lambda x: x.label() in tagset)):
                #     sentence_tags.append(st.label())
                #
                sentence_tags = [st.label() for st in s.subtrees(filter=lambda x: x.label() in tagset)]

                logging.debug(sentence_tags)
                syntags_lists[tagset_i] += " " + " ".join(sentence_tags)

                logging.debug("tagset %s: %s" % (tagset_i, syntags_lists[tagset_i]))

            # count height of a tree
            syntactic_tree_heigt_list.append(float(s.height()))
            logging.debug("tree height: %s" % s.height())

            # count subtrees with height bigger then 2
            subtree_count = len([st for st in s.subtrees(filter=lambda x: x.height() > 2)])
            syntactic_subtree_count_list.append(subtree_count)
            logging.debug("syn. subtree count: %s" % subtree_count)
        else:
            for tagset_i, tagset in enumerate(tagsets):
                syntags_lists[tagset_i] = [""]

            syntactic_tree_heigt_list.append(0)
            syntactic_subtree_count_list.append(0)

    return syntags_lists , syntactic_tree_heigt_list, syntactic_subtree_count_list

def do_extract_postags(text_list, tagger="SPACY"):
    text_postags_list = []      # tuple representation: (word, POS)
    text_postags_str_list = []  # string representation of POS tags only
    text_postags_extended_str_list = []  # string representation of POS tags only

    if tagger == "SPACY":
        postag_extraction_method = extract_postags_from_text_spacy
    else:
        postag_extraction_method = extract_postags_from_text_nltk

    print ("extract postags...")
    for txt in text_list:
        tokens_postagged, tokens_postagged_str, tokens_postagged_extended_str = postag_extraction_method(txt)
        text_postags_list.append (tokens_postagged)
        text_postags_str_list.append (tokens_postagged_str)
        text_postags_extended_str_list.append (tokens_postagged_extended_str)

    return text_postags_list, text_postags_str_list, text_postags_extended_str_list

def extract_postags_from_text_nltk(txt):

    # extract POS-Tags
    tokens = nltk.word_tokenize(txt)

    # print tokens
    tokens_postagged = nltk.pos_tag(tokens)

    tokens_postagged_str = " ".join([pos for (word, pos) in tokens_postagged])

    return tokens_postagged, tokens_postagged_str, None

def extract_postags_from_text_spacy(txt):

    # extract POS-Tags
    tokens = SPACY_NLP(txt)

    tokens_postagged = " ".join(["('%s','%s')" % (token.text, token.pos_) for token in tokens])
    tokens_postagged = "[" + tokens_postagged + "]"
    tokens_postagged_str = " ".join([token.pos_ for token in tokens])
    tokens_postagged_extended_str = " ".join([token.tag_ for token in tokens])

    logging.debug(tokens_postagged_str)

    return tokens_postagged, tokens_postagged_str, tokens_postagged_extended_str

def extract_wordpos_tuples_from_text(txt):

    wordpos_tuple, postags_string, x = extract_postags_from_text_nltk(txt)
    return wordpos_tuple

def extract_postags_string_from_text(txt):

    wordpos_tuple, postags_string, x = extract_postags_from_text_nltk(txt)
    return postags_string

def do_extract_special_postag_rel_frequency(postags_string_list):

    NN_freqs = []
    NN_freqs_rel_norm = []
    VB_freqs = []
    VB_freqs_rel_norm = []
    JJ_freqs = []
    JJ_freqs_rel_norm = []
    RB_freqs = []
    RB_freqs_rel_norm = []
    MD_freqs = []
    MD_freqs_rel_norm = []

    for postags_string in postags_string_list:
        (NN_freq, NN_freq_rel_norm), \
        (VB_freq, VB_freq_rel_norm), \
        (JJ_freq, JJ_freq_rel_norm), \
        (RB_freq, RB_freq_rel_norm), \
        (MD_freq, MD_freq_rel_norm) = extract_special_postag_rel_frequencies(postags_string)

        NN_freqs.append(NN_freq)
        NN_freqs_rel_norm.append(NN_freq_rel_norm)
        VB_freqs.append(VB_freq)
        VB_freqs_rel_norm.append(VB_freq_rel_norm)
        JJ_freqs.append(JJ_freq)
        JJ_freqs_rel_norm.append(JJ_freq_rel_norm)
        RB_freqs.append(RB_freq)
        RB_freqs_rel_norm.append(RB_freq_rel_norm)
        MD_freqs.append(MD_freq)
        MD_freqs_rel_norm.append(MD_freq_rel_norm)

    return  [(NN_freqs, NN_freqs_rel_norm),
            (VB_freqs, VB_freqs_rel_norm),
            (JJ_freqs, JJ_freqs_rel_norm),
            (RB_freqs, RB_freqs_rel_norm),
            (MD_freqs, MD_freqs_rel_norm)]

def extract_special_postag_rel_frequencies(txt):
    NN_freq, NN_freq_rel_norm = extract_postag_rel_frequency(txt, "NN")  # Nomen
    VB_freq, VB_freq_rel_norm = extract_postag_rel_frequency(txt, "VB")  # Verb

    JJ_freq, JJ_freq_rel_norm = extract_postag_rel_frequency(txt, "JJ")  # Nomen  # adjektiv
    RB_freq, RB_freq_rel_norm = extract_postag_rel_frequency(txt, "RB")  # Nomen  # adverb
    MD_freq, MD_freq_rel_norm = extract_postag_rel_frequency(txt, "MD")  # Modal

    return  [(NN_freq, NN_freq_rel_norm),
            (VB_freq, VB_freq_rel_norm),
            (JJ_freq, JJ_freq_rel_norm),
            (RB_freq, RB_freq_rel_norm),
            (MD_freq, MD_freq_rel_norm)]

def extract_postag_rel_frequency(txt, postag):
    postags_string = extract_postags_string_from_text(txt)
    wordlist = postags_string.split()

    postag_freq = len([w for w in wordlist if w.startswith(postag)])                            # count postags
    postag_freq_rel_norm = len([w for w in wordlist if w.startswith(postag)]) / len(wordlist)   # normalize count

    return postag_freq, postag_freq_rel_norm