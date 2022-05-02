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


def target_encoder(df, features, target, hierarchy=False):

    # helpers
    monotone_lambda = lambda x: x.n / (x.n + x.m)
    bayesian_enc = lambda x: x.l * x.ny / x.n + (1-x.l) * x.ny_prior / x.n_prior

    # features and priors lists
    if not hierarhcy:
        priors = len(features) * ['dataset']
    else:
        priors = ['dataset'] + features[:-1]

    # full dataset category
    df.insert(0, 'dataset', 1)

    agg_conf = {
        'ny': (target, 'sum'),
        'n': (target, 'count'),
        'sigma': (target, 'var')
    }

    encodings = {}
    for prior, feature in zip(priors, features):
        df_prior = df.groupby(prior, as_index=False).agg(**agg_conf)
        df_post = df.groupby([prior, feature], as_index=False).agg(**agg_conf)

        df_enc = df_post.merge(df_prior, on=prior, suffixes=('', '_prior'))

        df_enc['m'] = df_enc['sigma'] / df_enc['sigma_prior']
        df_enc['l'] = df_enc.apply(monotone_lambda, axis=1)

        df_enc['value'] = df_enc.apply(bayesian_enc, axis=1)
        df_enc.rename(columns={feature: 'key'})

        mapping = {x.key: x.value for x in df_enc.itertuples()}
        mapping = defaultdict(lambda: 0, mapping)

        encodings[feature] = lambda x: [mapping[i] for i in x]

    encoder = lambda x: encodings[feature](x)

    return encoder
