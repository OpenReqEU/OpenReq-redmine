# issue-weight

This service takes a french issue report and predict it's priority. The priority is a weight beetween -2 and 2 that can be used by a customer service to sort the demands automatically.

This service was created as a result of the OpenReq project funded by the European Union Horizon 2020 Research and Innovation programme under grant agreement No 732463.

## usage

\[WIP]

The endpoint `/issue_weighting` takes a json object with two properties:

- `Title` the title of the client issue report
- `Body` the body of the client issue report

More informations can be found in the [swager documentation](swagger.yaml).

## build & run

You can re-train the model on your datasets by replacing the files in `code/URMiner/data_repository/labeled_data`.

```bash
docker build -t issue-weight:1 .
docker run -p 9704:9704 --name issue-weight --rm issue-weight:1
```

## about

This repository is a fork of [openReq/ri-analytics-rationale-miner](https://github.com/OpenReqEU/ri-analytics-rationale-miner).

This repository uses spacy for preprocessing and scikit-learn to create nayve_bayes regressors, the result of the regressors determine the final weight.

## License

Free use of this software is granted under the terms of the EPL version 2 (EPL2.0).