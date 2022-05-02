import numpy as np
import pandas as pd
import pdb

from collections import defaultdict

def ordinal_encoder(x, top_n=None):
    
    categories, counts = np.unique(x, return_counts=True)
    data = {
        'key': categories,
        'counts': counts
    }
    df = pd.DataFrame(data)

    df.sort_values('counts', ascending=False, inplace=True)
    if top_n is None:
        top_n = len(categories)
    df = df.head(top_n)

    df['value'] = np.arange(len(df))

    mapping = {x.key: x.value for x in df.itertuples()}
    mapping = defaultdict(lambda: -1, mapping)

    encoder = lambda x: [mapping[i] for i in x]

    return encoder

