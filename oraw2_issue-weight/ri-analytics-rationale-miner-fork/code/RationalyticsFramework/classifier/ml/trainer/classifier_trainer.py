"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of the Rationalytics framework and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

import datetime
import inspect
import json
import logging
import os.path
import pathlib

import joblib
from RationalyticsFramework.classifier.data.data_selectors.data_selector_basic import \
    DataPreSelector, \
    DataSelector, \
    TypedDataSelector  # , PENNClauseTagsExtractor, PENNPhraseTagsExtractor
from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion

from RationalyticsFramework.classifier.data.preprocessors.preprocessor_basic import DenseTransformer
from RationalyticsFramework.classifier.ml.configs.classifier_configs_feature import F_TYPE_NGRAM, F_TYPE_NUMERIC
from RationalyticsFramework.classifier.ml.results.classifier_results_writer import y_array_write2csv, CSV_SEPARATOR, prf1_2csv

CLF_MODEL_FILE_SUFFIX = ".model.pkl"
CLF_CFG_FILE_SUFFIX = ".cfg.pkl"
ABOUT_FILE_SUFFIX = ".about.json"

def _safe_check_bool_cfg_key(cfg, key):
    try:
        value = getattr(cfg, key)
        assert type(value) == bool
    except AttributeError:
        logging.error("No attribute %s.." % key)
        value = False

    return value

def make_data_processing_pipeline(classifier_cfg, is_evaluation=False):

    # preprocessing_pipeline_list = []
    feature_union_list = []
    funion_union_weights_dict = {}

    feature_id_list = classifier_cfg.get_feature_cfg_id_list()

    if is_evaluation:
        tps = classifier_cfg.get_truthset_handler().get_text_parts_selector_strategy()
    # else:
    #     tps = TextPartsSelector()

    for f_id in feature_id_list:
        f_uid = classifier_cfg.get_feature_unique_id(f_id)
        f_type = classifier_cfg.get_feature_type(f_id)
        column = classifier_cfg.get_feature_data_column(f_id)

        #
        # feature extraction pipeline
        feature_extraction_pipeline = []
        item_selector_class = classifier_cfg.get_item_selector(f_id)
        if item_selector_class:
            logging.debug("Add item selector for %s: %s" % (f_uid, item_selector_class.__name__))
            item_selector_params = classifier_cfg.get_item_selector_params(f_id)
            ctor_args = inspect.getfullargspec(item_selector_class.__init__)[0]

            if DataPreSelector.CTOR_P_TEXT_PARTS_SELECTOR in ctor_args:
                item_selector_params.update({DataPreSelector.CTOR_P_TEXT_PARTS_SELECTOR : tps})
            if DataPreSelector.CTOR_P_IS_EVALUATION in ctor_args:
                item_selector_params.update({DataPreSelector.CTOR_P_IS_EVALUATION: is_evaluation})

            item_selector = item_selector_class(**item_selector_params)
            feature_extraction_pipeline.append(('%s_preprocess_selector' % f_uid, item_selector))

        preprocessor_class = classifier_cfg.get_feature_preprocessor(f_id)
        if preprocessor_class:
            logging.debug("Add preprocessor for %s: %s" % (f_uid, preprocessor_class.__name__))
            preprocessor_params = classifier_cfg.get_feature_preprocessor_params(f_id)

            # Create a preprocessor instance
            preprocessor_instance = preprocessor_class(**preprocessor_params)

            # Get the name of the column representing the preprocessed text
            preprocessed_col_name = preprocessor_instance.get_processed_col_name()

            # include into preprocessing pipeline list
            feature_extraction_pipeline.append(
                ("%s_preprocessor" % f_uid, preprocessor_instance))
            # #
            # # select preprocessed input using a default data selector, that select a single data key
            # logging.debug("Add default selector for %s: %s" % (f_uid, DataPreSelector.__name__))
            # if f_type == F_TYPE_NGRAM:
            #     postprocess_selector = DataSelector(**{DataSelector.CTOR_P_KEY : preprocessed_col_name})
            # elif f_type == F_TYPE_NUMERIC:
            #     postprocess_selector = TypedDataSelector(**{TypedDataSelector.CTOR_P_KEY: preprocessed_col_name,
            #                                                 TypedDataSelector.CTOR_P_RESHAPE: True,
            #                                                 TypedDataSelector.CTOR_P_VALUE_TYPE: float})
            # else:
            #     raise Exception("cant set postprocessor.. unknown feature type")

        else:
            preprocessed_col_name = column

        #
        # default data selector
        logging.debug("Add default selector for %s: %s" % (f_uid, DataPreSelector.__name__))
        if f_type == F_TYPE_NGRAM:
            postprocess_selector = DataSelector(**{DataSelector.CTOR_P_KEY: preprocessed_col_name})
        elif f_type == F_TYPE_NUMERIC:
            postprocess_selector = TypedDataSelector(**{TypedDataSelector.CTOR_P_KEY: preprocessed_col_name,
                                                        TypedDataSelector.CTOR_P_RESHAPE: True,
                                                        TypedDataSelector.CTOR_P_VALUE_TYPE: float})
        else:
            raise Exception("cant set postprocessor.. unknown feature type")
        feature_extraction_pipeline.append(('%s_default_postprocess_selector' % f_uid, postprocess_selector))


        feature_extractor_class = classifier_cfg.get_feature_extractor(f_id)
        if feature_extractor_class:
            logging.debug("Add feature extractor for %s: %s" % (f_uid, feature_extractor_class.__name__))
            feature_extractor_params = classifier_cfg.get_feature_extractor_params(f_id)
            if feature_extractor_params is None:
                feature_extractor_params = {}
            feature_extractor_instance = feature_extractor_class(**feature_extractor_params)
            feature_extraction_pipeline.append(('%s_extractor' % f_uid, feature_extractor_instance))

        feature_normalizer_class = classifier_cfg.get_feature_normalizer(f_id)
        if feature_normalizer_class:
            logging.debug("Add feature normalizer for %s: %s" % (f_uid, feature_normalizer_class.__name__))
            feature_normalizer_params = classifier_cfg.get_feature_normalizer_params(f_id)
            if feature_normalizer_params is None:
                feature_normalizer_params = {}
            feature_normalizer_instance = feature_normalizer_class(**feature_normalizer_params)
            feature_extraction_pipeline.append(('%s_normalizer' % f_uid, feature_normalizer_instance))

        feature_dim_reducer_class = classifier_cfg.get_feature_dim_reducer(f_id)
        if feature_dim_reducer_class:
            logging.debug("Add feature dim reducer for %s: %s" % (f_uid, feature_dim_reducer_class.__name__))
            feature_dim_reducer_params = classifier_cfg.get_feature_dim_reducer_params(f_id)
            if feature_dim_reducer_params is None:
                feature_dim_reducer_params = {}
            feature_dim_reducer_instance = feature_dim_reducer_class(**feature_dim_reducer_params)
            feature_extraction_pipeline.append(('%s_dim_reducer' % f_uid, feature_dim_reducer_instance))

        feature_union_list.append(
            (f_uid, Pipeline(feature_extraction_pipeline)),
        )

    #
    # create final pipeline
    # preprocessing_pipeline = None
    # if len(preprocessing_pipeline_list):
    #     preprocessing_pipeline = Pipeline(preprocessing_pipeline_list)

    feature_union = FeatureUnion(
        transformer_list=feature_union_list,
        transformer_weights=funion_union_weights_dict,
    )

    final_pipeline = Pipeline([
        # ("preprocessor", preprocessing_pipeline),
        ("feature_union", feature_union),
    ])

    return final_pipeline


def make_classifier_training_pipeline(classifier_cfg, classifier_algo_cfg, dim_reducer=None, is_evaluation=False):
    preprocessing_pipeline = make_data_processing_pipeline(classifier_cfg=classifier_cfg, is_evaluation=is_evaluation)

    # any classifier specific transformations to consider?
    algo_cfg = classifier_algo_cfg #
    classifier_pipeline_list = []
    if algo_cfg.dense_matrix_required:
        classifier_pipeline_list.append(("dense_transformer", DenseTransformer()))

    if dim_reducer:
        logging.debug("Add dim_reducer %s" % dim_reducer.__class__.__name__)
        # include dense_transformer, if already not included (e.g., needed for PCA)
        if not classifier_cfg.dense_matrix_required:
            classifier_pipeline_list.append(("dense_transformer", DenseTransformer()))

        #todo do we need this final reducer?
        classifier_pipeline_list.append(('reduce_dim', dim_reducer))

    classifier_pipeline_list.append((algo_cfg.algorithm_code, algo_cfg.algorithm_instance))
    classifier_pipeline = Pipeline(classifier_pipeline_list)

    training_pipeline = Pipeline([
        ("data_preprocessor", preprocessing_pipeline),
        ("classifier", classifier_pipeline),
    ])

    return training_pipeline

def check_model_file_exists(base_filename):
    model_file_path = base_filename + CLF_MODEL_FILE_SUFFIX
    file_exists = os.path.isfile(model_file_path)

    return file_exists

def train_and_persist_classifier_model(classifier_cfg, label_column, base_filename, max_items_per_class=None):

    # get the classifier algorithm config
    algo_cfg = classifier_cfg.get_default_classifier_algorithm()

    # make pipeline
    training_pipeline = make_classifier_training_pipeline(classifier_cfg=classifier_cfg,
                                        classifier_algo_cfg=algo_cfg,
                                        dim_reducer=None)

    # get data classifier_cfg
    logging.debug("Select training/testing data..")
    data_cfg = classifier_cfg.get_data_sample_config(label_col=label_column, max_items_per_class=max_items_per_class)
    df = data_cfg.data_wrapper.compiled_df()
    logging.debug("training user comments dataframe shape: %s", df.shape)

    X = data_cfg.data_wrapper.compiled_df()
    y = data_cfg.data_wrapper.get_user_comments().get_df()[data_cfg.label_column]

    logging.debug("Train classifier...")
    model = training_pipeline.fit(X, y)

    out_filename = base_filename
    out_filename_model = out_filename + CLF_MODEL_FILE_SUFFIX
    out_filename_configuration = out_filename + CLF_CFG_FILE_SUFFIX
    out_filename_about = out_filename + ABOUT_FILE_SUFFIX

    # persist model
    logging.debug("Persist classification model to file: %s" % out_filename_model)
    joblib.dump(model, out_filename_model)

    # persist configuration
    logging.debug("Persist classification config to file: %s" % out_filename_configuration)
    joblib.dump(classifier_cfg, out_filename_configuration)

    #
    # create about dict
    info_dict = {}

    import subprocess
    try:
        git_hash = subprocess.check_output(["git", "describe", "--always"])
        info_dict["git_hash"] = str(git_hash, "UTF-8").strip()
    except:
        logging.debug("git hash extraction failed")

    info_dict["label_column"] = label_column
    info_dict["algorithm_code"] = algo_cfg.algorithm_code
    info_dict["date_time"] = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")

    # persist about dict
    # persist configuration
    logging.debug("Persist about json to file: %s" % out_filename_about)
    with open(out_filename_about, 'w') as fi:
        json.dump(info_dict, fi)

    return model


def load_classifier_model(base_filename):
    base_filename = base_filename
    filename_model = base_filename + CLF_MODEL_FILE_SUFFIX
    # filename_configuration = base_filename + CLF_CFG_FILE_SUFFIX

    clf_model = joblib.load(filename_model)
    # clf_cfg = joblib.load(filename_configuration)

    return clf_model


def _evaluate_classifier(classifier_cfg, labels, algorithm_codes, tag, random_state, output_dir="results", dim_reducer=None, max_items_per_label=None):
    for l in labels:
        logging.debug("Evaluation for label %s" % l)
        # get data classifier_cfg
        data_cfg = classifier_cfg.get_data_sample_config(label_col=l, max_items_per_class=max_items_per_label)
        compiled_df = data_cfg.data_wrapper.compiled_df()

        logging.debug("user comments df shape: %s", compiled_df.shape)

        df_X = compiled_df
        #todo move y values in the data wrapper in a separate df?
        df_y = compiled_df[data_cfg.label_column]

        # write y-array
        y_true_array_tag = tag + "_yTRUE"
        y_array_write2csv(df_y, data_cfg=data_cfg, classifier_algo_code=None, tag=y_true_array_tag)

        for algo in algorithm_codes:
            logging.debug("Evaluation for algorithm %s" % algo)
            # select classifier & feature configurations

            algo_cfg = classifier_cfg.get_classifier_algorithm_config(algo)

            # make pipeline
            pipe = make_classifier_training_pipeline(classifier_cfg=classifier_cfg,
                                                     classifier_algo_cfg=algo_cfg,
                                                     dim_reducer=dim_reducer,
                                                     is_evaluation=True)

            # do cross validation
            ccv = classifier_cross_validator(clf=pipe,
                                             X=df_X,
                                             y=df_y,
                                             shuffle=False,  # data is already shuffled
                                              random_state=random_state)

            # write output to csv
            df_results = prf1_2csv(df_y, ccv["y_predicted"], data_cfg, classifier_cfg, classifier_algo_code=algo, tag=tag)
            filename = "%s/results.csv" % output_dir
            _save_results_to_csv(df_results, filename)

            # write y-array
            df_yarray = y_array_write2csv(ccv["y_predicted"], data_cfg, classifier_algo_code=algo, tag=tag)
            filename = "%s/yarrays.csv" % output_dir
            _save_results_to_csv(df_yarray, filename)

def _save_results_to_csv(df, out_file):
    file = pathlib.Path(out_file)

    if file.exists():
        with open(out_file, 'a') as f:
            df.to_csv(f, header=False, sep=CSV_SEPARATOR, index=False, encoding='UTF-8')
    else:
        df.to_csv(out_file, header=True, sep=CSV_SEPARATOR, index=False, encoding='UTF-8')

def classifier_cross_validator(clf, X, y, shuffle, random_state, n_splits=10):
    logging.debug("classifier cross-validation ...")

    to_return = {}

    # prepare the cross validator
    cv = model_selection.KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    to_return["cv"] = cv

    # conduct a cross-validation and
    predicted = model_selection.cross_val_predict(
        estimator=clf, X=X, y=y, cv=cv, n_jobs=-1, verbose=1)
    to_return["y_predicted"] = predicted

    print(classification_report(y, predicted))

    # fraction of correctly classified samples
    accuracy = accuracy_score(y, predicted)
    print(accuracy)
    to_return["accurracy"] = accuracy
    logging.debug("Accurracy: %s", accuracy)

    # confustion matrix
    cm = confusion_matrix(y, predicted)
    print(cm)
    to_return["confusion_matrix"] = cm

    return to_return

