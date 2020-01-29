import pandas as pd
import argparse
import urllib.request
import json
from sklearn.metrics import classification_report

# arg parse
parser = argparse.ArgumentParser(description='Evaluate the binary classifier generated.')
parser.add_argument('-path', help='csv testing data set containing at least the columns Title Body and Stance')
parser.add_argument('-url', nargs='?', help='endpoint address')
args = parser.parse_args()

# init data
df_test = pd.read_csv(args.path)
predictions = {
    'anomaly': [],
    'urgence': []
}

def rowToData(row):
    return json.dumps({
        'Title': row['subject'], 
        'Body': row['description'], 
        'Stance': 0
        }).encode('utf8')

# fetch all
for idx, row in df_test.iterrows():
    req = urllib.request.Request(
        args.url,
        data=rowToData(row),
        headers={'content-type': 'application/json'})

    data = json.loads(urllib.request.urlopen(req).read().decode('utf8'))['data']

    predictions['anomaly'].append('Anomaly' if data['isAnomalyRatio'] > 0.5 else 'Demand')
    if data['isLowPrioRatio'] > data['isAvgPrioRatio'] and data['isLowPrioRatio'] > data['isHighPrioRatio']:
        predictions['urgence'].append('Basse')
    elif data['isHighPrioRatio'] > data['isAvgPrioRatio'] and data['isHighPrioRatio'] > data['isLowPrioRatio']:
        predictions['urgence'].append('Haute')
    else:
        predictions['urgence'].append('Normale')

import pickle
pickle.dump([df_test, predictions], open('preds.pkl', 'wb'))

print("anomaly", classification_report(df_test['tracker'], predictions['anomaly']))
print("urgence", classification_report(df_test['urgence'], predictions['urgence']))