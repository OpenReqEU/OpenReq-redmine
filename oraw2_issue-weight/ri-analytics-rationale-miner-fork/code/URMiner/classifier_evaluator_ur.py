import logging

from classifier.ml.configs.classifier_configs import get_baseline_classifier_algorithm_codes
from classifier.ml.trainer.classifier_trainer import _evaluate_classifier
#from classifier_ur.data import UR_MAIN_LABEL_COLUMNS
from classifier_ur.ml.classifier_review import URBaselineReviewClassifierConfig
from classifier_ur.ml.classifier_sentence import URBaselineSentenceClassifierConfig
from classifier_ur.ml.classifier_ur import URClassifier

def evaluate_classifier_wrapper():
    logging.basicConfig(level=logging.DEBUG)

    # cfg = URBaselineSentenceClassifierConfig()
    cfg = URBaselineReviewClassifierConfig()
    # cfg = URAllFeatureTypesClassifierConfig(th)
    cfg.set_default_classifier_algorithm_via_code("nb")

    label_column = "HasJustification"

    ur_clf = URClassifier(cfg, label_column, "baseline_review")
    # result = ur_clf.predict("This is really bad", "I really don't like the new UI! The buttons freeze from time to time", 2)
    result = ur_clf.predict_ur("I love it!", "The tool is just super cool! Good job Company!", 5)

    print (result)

#
# Debug
#

def evaluate_DEBUG(cfg, label, algo, max_items_per_label=None):
    #todo edit
    labels = [label]#UR_MAIN_LABEL_COLUMNS
    algo_codes = [algo]
    tag = "DEBUG"

    _evaluate_classifier(cfg, labels=labels, algorithm_codes=algo_codes, tag=tag, random_state=2,
                         dim_reducer=None,
                         max_items_per_label=max_items_per_label)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    evaluate_classifier_wrapper()