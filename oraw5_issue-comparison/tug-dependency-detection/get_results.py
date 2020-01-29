import json

import pandas as pd

df = pd.read_csv('../aw-closest-text/data/ref_subset.csv',
    converters={'rels': lambda d: json.loads(d)})
df_aw = pd.read_json('../aw-closest-text/result_subset.json')
df_or = pd.read_json('../tug-dependency-detection/result_subset.json')

aw_good, aw_miss, or_good, or_miss = (0,0,0,0)

for idx, vals in df.iterrows():
    aw_pred = df_aw.loc[vals['id'] == df_aw['id']]['closest_ids'].values[0]
    or_pred = df_or.loc[vals['id'] == df_or['id']]['predictions'].values[0]

    for val in vals['rels']:
        if vals['id'] in df_or.loc[val == df_or['id']]['predictions'].values[0]:
            or_pred += [val]

        if val in aw_pred:
            aw_good += 1
        else:
            print('aw_miss', vals['id'], val)
            aw_miss += 1
    
        if val in or_pred:
            or_good += 1
        else:
            print('or_miss', vals['id'], val)
            or_miss += 1

    print(vals['rels'], aw_pred, or_pred)

print(aw_good, aw_miss, or_good, or_miss)
print(aw_good / (aw_good + aw_miss), or_good / (or_good + or_miss))