import os
import sys

from URMiner.consts_ur import ROOT_DIR

#UR_REVIEW_CSV_FILE=UR_SENTENCE_CSV_FILE= "%s/%s" % (ROOT_DIR, "data_repository/labeled_data/ticket_san_dataset_for_rational_miner_train.csv")
UR_REVIEW_CSV_FILE=UR_SENTENCE_CSV_FILE= "%s/%s" % (ROOT_DIR, "data_repository/labeled_data/oraw1_15k_train2.csv")

COL_IS_ANOMALY="IsAnomalie"
COL_IS_LOW_PRIORITY="IsLowPriority"
COL_IS_AVG_PRIORITY="IsAvgPriority"
COL_IS_HIGH_PRIORITY="IsHighPriority"

COL_TEXT_BODY = "Body"
COL_TEXT_TITLE = "Title"
COL_RATING="Stance"
COL_INDEX_SENTENCE="RelativeOrder"