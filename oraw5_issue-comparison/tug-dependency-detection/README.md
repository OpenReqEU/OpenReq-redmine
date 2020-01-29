# Requirement Dependency Detection Service [![EPL 2.0](https://img.shields.io/badge/License-EPL%202.0-blue.svg)](https://www.eclipse.org/legal/epl-2.0/)

This service was created as a result of the OpenReq project funded by the European Union Horizon 2020 Research and Innovation programme under grant agreement No 732463.
This project uses the [Connexion](https://github.com/zalando/connexion) library on top of Flask.

## Technical description
### What does the service do
This service analyzes a given set of requirements and finds requirement pairs which potentially depend on each other
by measuring the similarity between the both requirements of the pair.
The recommended requirement pairs are those whose similarity is not too high but also not too low.
The implemented approach is a very basic unsupervised technique which can only detect dependencies on a content-based level.
A more sophisticated supervised approach which requires training data and goes beyond this basic concept is explained in the OpenReq paper of Atas et al.:
https://ieeexplore.ieee.org/abstract/document/8609673

### Which technologies are used
This service requires Python 3.7.0+

- Docker (-> https://www.docker.com/)
- Flask Connexion (-> https://github.com/zalando/connexion/)
- python-dateutil (-> https://pypi.org/project/python-dateutil/)
- scikit-learn (-> https://scikit-learn.org/)
- SciPy (-> https://www.scipy.org/)
- NumPy (-> http://www.numpy.org/)
- Flask (-> https://github.com/pallets/flask/)
- NLTK (-> https://github.com/nltk/nltk/)
- Pattern Library (-> https://github.com/clips/pattern)
- Werkzeug (-> http://werkzeug.pocoo.org/)
- Setuptools (-> https://pypi.org/project/setuptools/)


### How to install it
To run the server and to install all dependencies, please execute the following commands from the project root directory:

```
pip3 install -r requirements.txt
python3 -m application
```

To launch the integration tests, use tox:
```
sudo pip install tox
tox
```

### Running with Docker

To run the server on a Docker container, please execute the following commands from the project root directory:

```bash
# building the image
docker build -t application .

# starting up a container
docker run -p 9007:9007 application
```

## How to use it (high-level description)

Once the server is running, open your browser and call the following URL to see the API documentation:

```
http://217.172.12.199:9007/v1/ui/
```

The Swagger definition lives here:

```
http://217.172.12.199:9007/v1/swagger.json
```

[Rendered Documentation](https://api.openreq.eu/#/services/tug-dependency-detection)

### Notes for developers
None.

### Sources
None.

### How to contribute
See OpenReq project contribution [Guidlines](https://github.com/OpenReqEU/OpenReq/blob/master/CONTRIBUTING.md "Guidlines")

## License
Free use of this software is granted under the terms of the EPL version 2 ([EPL2.0](https://www.eclipse.org/legal/epl-2.0/)).
