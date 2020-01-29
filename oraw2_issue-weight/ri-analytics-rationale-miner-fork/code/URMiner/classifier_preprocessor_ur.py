from classifier_ur.transformers.classifier_preprocessor import TextPreprocessor, TextLengthExtractor


#
# Methods for caching of the preprocessing/feature extractors
#

def _preprocess_truthset_sentence():
    df = get_truthset_sentence()
    tp = TextPreprocessor("Value", True, True, True)
    df = tp.transform(df)
    tle = TextLengthExtractor("Value")
    df = tle.transform(df)

    save_truthset_sentence(df)


def _preprocess_truthset_review(value_col):
    df = get_truthset_review()
    tp = TextPreprocessor(value_col, True, True, True)
    df = tp.transform(df)
    tle = TextLengthExtractor(value_col)
    df = tle.transform(df)

    save_truthset_review(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    _preprocess_truthset_sentence()

    # # ur_data = TruthsetHandlers.ur_sentence()
    # ur_data = TruthsetHandlers.ur_review()
    # df = ur_data._get_truthset()
    #
    # text_col_name = "Value"
    # transformer = [
    #     TextPreprocessor(text_col_name, True, True, True),
    #     # TextLengthExtractor(col_name),
    #     # POSTagsExtractor(text_col_name),
    #     # NumericValueExtractor("Rating"),
    #     # ScaledNumericValueExtractor("Rating")
    #     PENNClauseTagsExtractor("Value"),
    #     PENNPhraseTagsExtractor("Value")
    # ]
    #
    # for t in transformer:
    #     df_ = t.transform(df)
    #
    #     # _preprocess_truthset_sentence()
    #     # _preprocess_truthset_review("Title")

