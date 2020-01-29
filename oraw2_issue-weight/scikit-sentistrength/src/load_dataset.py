import pandas as pd
import numpy as np
from sentistrength import getSentiment

def load_dataset(name, test=False):
    sufix = "_test.csv" if test else "_train.csv"
    df = pd.read_csv(name + sufix).astype(str)
    df['Text'] = df['subject'] + df['description']
    df['Senti_pos'], df['Senti_neg'] = df.apply(
        lambda row: getSentiment(row['Text']), 
        axis=1, 
        result_type='expand')

    x = df[['Text', 'Senti_pos', 'Senti_neg']].values.tolist()
    y = df[['tracker','urgence']].values.tolist()

    return x, y