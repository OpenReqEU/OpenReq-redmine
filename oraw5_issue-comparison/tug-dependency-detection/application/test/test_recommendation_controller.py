# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from application.models.requirement import Requirement  # noqa: E501
from application.models.requirement_popularity import RequirementPopularity  # noqa: E501
from application.test import BaseTestCase


class TestRecommendationController(BaseTestCase):
    """RecommendationController integration test stubs"""

    def test_compute_popularity(self):
        """Test case for compute_popularity

        Retrieve a list with values for given set of requirements indicating their popularity for the crowd on twitter.
        """
        body = [Requirement()]
        response = self.client.open(
            '/v1/popularity',
            method='POST',
            data=json.dumps(body),
            content_type='application/json')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
