"""
Copyright (C) 2017-2018 Zijad Kurtanovic <kurtanovic@informatik.uni-hamburg.de>

This file is part of the Rationalytics framework and subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
"""

from abc import ABC, abstractmethod

class DataWrapper(object):
    def __init__(self, user_comments, commented_items=None, replies=None):

        assert (user_comments != None)

        self._commented_items = commented_items
        self._user_comments = user_comments
        self._replies = replies

        # needed for scikit pipelines to work
        # self.X = self._user_comments.get_df().shape
        self.shape = self._user_comments.get_df().shape

    def get_commented_items(self):
        return self._commented_items

    def get_user_comments(self):
        return self._user_comments

    def get_replies(self):
        return self._replies

    def compiled_df(self):
        df = self._user_comments.get_df()

        if self._commented_items is not None:
            df = self._commented_items.get_df()
            #todo concatenate this df with the user comments one
        if self._replies is not None:
            # todo concatenate this df with the user comments one
            df = self._replies.get_df()

        return df

class ADataWrapper(ABC):
    @abstractmethod
    def get_df(self):
        pass

class CommentedItems(ADataWrapper):
    def __init__(self, df):
        self._df = df

    def get_df(self):
        return self._df

    def get_title(self, id):
        return self._df[id]

class UserComments(ADataWrapper):
    def __init__(self, df, text_parts_selector=None):
        self._parts_selector = text_parts_selector
        self._df = df

    def get_df(self):
        return self._df

    def get_text_parts_selector(self):
        return self._parts_selector

class Replies(ADataWrapper):
    def __init__(self, df):
        self._df = df

    def get_df(self):
        return self._df

    def get_reply(self, comment_id):
        return self._df[comment_id]