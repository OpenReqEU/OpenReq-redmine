"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of the Rationalytics framework and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

ANALOGY_MARKER = ["as a", "just as", "comes from the same"]
ANTI_THESIS_MARKER = ["although", "even while", "on the other hand"]
CAUSE_MARKER = ["because", "as a result", "which in turn"]
CONCESSION_MARKER = ["despite", "regardless of", "even if"]
REASON_MARKER = ["because", "because it is", "to find a way"]

def _extract_marker_count(txt_list, marker_list):
    marker_output = []
    for txt in txt_list:
        tmp = _extract_marker_count_from_text(txt, marker_list)
        marker_output.append(tmp)
    return marker_output

def _extract_marker_count_from_text(txt, marker_list):
    tmp = sum([1 if word in txt else 0 for word in marker_list])
    return tmp


def extract_analogy_marker_count(txt_list):
    return _extract_marker_count(txt_list, ANALOGY_MARKER)

def extract_antithesis_marker_count(txt_list):
    return _extract_marker_count(txt_list, ANTI_THESIS_MARKER)

def extract_cause_marker_count(txt_list):
    return _extract_marker_count(txt_list, CAUSE_MARKER)

def extract_concession_marker_count(txt_list):
    return _extract_marker_count(txt_list, CONCESSION_MARKER)

def extract_reason_marker_count(txt_list):
    return _extract_marker_count(txt_list, REASON_MARKER)