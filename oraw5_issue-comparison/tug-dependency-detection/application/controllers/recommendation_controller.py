import connexion
import six
import logging
import json
import os
import csv
import functools
import itertools
from application.models.requirement import Requirement  # noqa: E501
from application.entities import requirement
from application.preprocessing import preprocessing
from application.unsupervised import svd
from application.util import helper
from pprint import pprint


_logger = logging.getLogger(__name__)
FILTER_CHARACTERS_AT_BEGINNING_OR_END = ['.', ',', '(', ')', '[', ']', '|', ':', ';']


def recommend_requirement_dependencies(body):  # noqa: E501
    """Retrieve a list with values for given set of requirements indicating their popularity for the crowd on twitter.

     # noqa: E501

    :param body: Requirement objects for which the social popularity should be measured
    :type body: list | bytes

    :rtype: List[Requirement]
    """

    response_list = []
    # TODO: introduce parameter to set language
    lang = "en"

    if connexion.request.is_json:
        content = connexion.request.get_json()
        assert isinstance(content, list)
        requs = [Requirement.from_dict(d) for d in content]  # noqa: E501

        requs = list(map(lambda r: requirement.Requirement(r.id, r.title, r.description, r.comments), requs))
        for r in requs:
            r.append_comments_to_description()

        requs = preprocessing.preprocess_requirements(requs,
                                                      enable_stemming=False,
                                                      lang=lang)

        requs = list(filter(lambda r: len(r.tokens()) > 0, requs))

        if len(requs) == 0:
            return []

        _logger.info("SVD...")

        # if len(requs) > 100:
        #     min_distance, max_distance = 0.2, 0.5
        #     k = 10
        # elif len(requs) > 50:
        #     min_distance, max_distance = 0.2, 0.6
        #     k = 8
        # elif len(requs) > 30:
        #     min_distance, max_distance = 0.2, 0.65
        #     k = 5
        # elif len(requs) > 10:
        #     min_distance, max_distance = 0.2, 0.7
        #     k = 3
        # elif len(requs) > 5:
        #     min_distance, max_distance = 0.2, 0.75
        #     k = 2
        # else:
        #     min_distance, max_distance = 0.2, 0.8
        #     k = 1
        # TODO
        k = 5 # number of recommendation
        min_distance, max_distance = 0.0, 0.8 # distance of the text

        predictions_map = svd.svd(requs, k=k, min_distance=min_distance, max_distance=max_distance)
        dependency_pairs = set()
        for subject_requirement, dependent_requirements in predictions_map.items():
            requ = Requirement.from_dict({
                "id": subject_requirement.id,
                "title": subject_requirement.title,
                "description": subject_requirement.description,
                "comments": subject_requirement.comments
            })
            rx = subject_requirement.id
            dependent_requirement_ids = list(set(map(lambda r: r.id, dependent_requirements)))
            all_undirected_pairs_of_subject_requirement = map(lambda ry: [(rx, ry), (ry, rx)], dependent_requirement_ids)
            dependency_pairs_of_subject_requirement = set(list(itertools.chain(*all_undirected_pairs_of_subject_requirement)))
            remaining_dependency_pairs_of_subject_requirement = dependency_pairs_of_subject_requirement - dependency_pairs
            dependency_pairs = dependency_pairs.union(remaining_dependency_pairs_of_subject_requirement)
            predictions = list(set(map(lambda t: t[0] if t[0] != rx else t[1], remaining_dependency_pairs_of_subject_requirement)))

            requ.predictions = predictions
            response_list += [requ]
            for dependent_requirement in dependent_requirements:
                print("{} -> {}".format(subject_requirement, dependent_requirement))

        """
        for idx, requ in enumerate(requirements):
            response_list.append(Requirement.from_dict({
                "id": requ.id,
                "title": requ.title,
                "description": requ.description
            }))
        """

    return response_list


def csv_reader(file_content):
    reader = csv.reader(file_content)
    content = []
    for row in reader:
        if row[0].strip() == "10820":
            break
        if row[-1].lower() == "def":
            content += [row[3]]
    return content


def perform_svd():
    enable_tagging = True
    max_distance = 0.6
    with open(os.path.join(helper.APP_PATH, "data", "requirements_en.json")) as f:
        requs = json.load(f)

    max_distance = 0.4
    with open(os.path.join(helper.APP_PATH, "data", "siemens_requirements_en.csv")) as f:
        enable_tagging = False
        plain_requirements = csv_reader(f)
        requs = []
        for (idx, description) in enumerate(plain_requirements):
            if idx > 400:
                break
            requs += [{
                'id': idx,
                'title': '',
                'description': description
            }]
    #print(json.dumps(requs))
    #import sys;sys.exit()

    #pprint(requs)
    requs = list(map(lambda r: Requirement.from_dict(r), requs))
    lang = "en"

    requs = list(map(lambda r: requirement.Requirement(r.id, r.title, r.description), requs))
    requs = preprocessing.preprocess_requirements(requs,
                                                  enable_pos_tagging=enable_tagging,
                                                  enable_lemmatization=enable_tagging,
                                                  enable_stemming=False,
                                                  lang=lang)

    _logger.info("SVD...")
    predictions_map = svd.svd(requs, k=3, max_distance=max_distance)
    for subject_requirement, similar_requirements in predictions_map.items():
        if len(similar_requirements) == 0:
            continue

        #print("-" * 80)
        #print(subject_requirement.description_tokens)
        for similar_requirement in similar_requirements:
            print("#{}: {} -> #{}: {}".format(subject_requirement.id,
                                              subject_requirement.description[:80],
                                              similar_requirement.id,
                                              similar_requirement.description[:80]))
            #print(similar_requirement.description_tokens)
