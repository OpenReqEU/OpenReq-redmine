"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of the Rationalytics framework and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

import subprocess
import sys

from consts import ROOT_DIR, LIBS_DIR


def do_extract_sentiments(sentences):
    """
    Perform sentiment analysis on a sentence using SentiStrength.
    Sentistrength has to be available in the directory above.
    """
    sentiment_pos_list = []
    sentiment_neg_list = []
    sentiment_norm_list = []

    for sentence in sentences:
        print (sentence)
        p = subprocess.Popen(
        [
            "java", "-jar", ROOT_DIR + LIBS_DIR + "/SentiStrength.jar",
            "stdin", "sentidata", ROOT_DIR + LIBS_DIR + "/SentStrength_Data/"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE
        )
        try:
            output = p.communicate(sentence.replace(" ", "+").encode('utf-8'))[0]
            sentiment = output.decode().strip().split(' ')
            # sentiment = str(output).encode('utf-8').strip().split(' ') #todo original line
            print (sentiment)

            sentiment_pos, sentiment_neg, sentiment_norm = \
                float(sentiment[0]), \
                float(sentiment[1]), \
                float(sentiment[0]) + float(sentiment[1]) + 5.0
            sentiment_pos_list.append(sentiment_pos)
            sentiment_neg_list.append(sentiment_neg)
            sentiment_norm_list.append(sentiment_norm)
        except:
            print ("error:")
            print (sys.exc_info())
            sentiment_pos_list.append(0)
            sentiment_neg_list.append(0)
            sentiment_norm_list.append(0)

    return sentiment_pos_list, sentiment_neg_list, sentiment_norm_list