# Issue, Alternative, Criteria, Decision, Rationale
import collections

DATA_ROOT = "/Users/zkey/Documents/MobisGIT/2017_RationalyticsFramework/DATA/"

GRANULARITY_COMMENT = "comment"
GRANULARITY_SENTENCE = "sentence"
GRANULARITY_LEVELS = [GRANULARITY_COMMENT, GRANULARITY_SENTENCE]

#todo do align id comment
COL_INDEX_COMMENT = "index_comment"
COL_INDEX_SENTENCE = "index_sentence"

CSV_SEPARATOR = ","

CONFIG_DATA_IDS = ["label_column", "classes", "max_items_per_class", "data_wrapper", "test_set"]
DataSampleConfig  = collections.namedtuple("data_config",
                                           [
        CONFIG_DATA_IDS[0],
        CONFIG_DATA_IDS[1],
        CONFIG_DATA_IDS[2],
        CONFIG_DATA_IDS[3],
        CONFIG_DATA_IDS[4],
        # CONFIG_DATA_IDS[5]
     ])
