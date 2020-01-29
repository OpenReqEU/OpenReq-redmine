# algorithm comparison results

## results

Raw results returned by both algorithms can be found here:

- [see result_subset.json](result_subset.json)
- [../aw-closest-text/result_subset.json](../aw-closest-text/result_subset.json)

They can be compared with the reference dataset: [../aw-closest-text/data/ref_subset.json](../aw-closest-text/data/ref_subset.json)

The OpenReq algorithm return unidirectional requirement by design, however for our use case we will treat the results as bidirectionnal.

Since AW's algorithm return always 5 proposals, his precision is'nt meaningfull, we will compare the algorithms based on the recall value (true positives / positives).

AW recall: 0.93 (76 true positive; 6 missing positive)

OR recall: 0.90 (74 true positive; 8 missing positive)

In regard of the size of the test dataset those algorithms performances are comparable.

## error cases

### tickets 15636 and 11572

Both tickets are explicitly about student account création but their length is quite different. Overall AW's algorithm is the most subject to miss predictions due to the text length, indeed the jaccard distance tends stay high for related text featuring a large vocabulary.

### tickets 9372 7730 6258

Those short tickets about printer issues were not identified as related by OR's algorithm, the cause is unclear, and does not sim related to the preprocessing.
