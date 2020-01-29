#!/usr/bin/env python3

import connexion
from application import encoder
from application.controllers import recommendation_controller


def main():
    app = connexion.App(__name__, specification_dir='./swagger/')
    app.app.json_encoder = encoder.JSONEncoder
    app.add_api('swagger.yaml', arguments={'title': 'OpenReq Requirement Dependency Recommendation Service'})
    app.run(port=9007)


def test_svd():
    print('Test SVD')
    recommendation_controller.perform_svd()


if __name__ == '__main__':
    main()
    #test_svd()
