import pandas as pd
from typing import Sequence

def _flatten_and_unique(sequences:Sequence[Sequence[int]]): # only 'utils-esque' function so just keeping here
    ''' get unique values from a sequence of sequences '''
    return sorted(set(i for sequence in sequences for i in sequence))

def _is_square_df(df:pd.DataFrame):
    if df.shape[0] != df.shape[1]:
        raise ValueError(f'DataFrame shape must be square. got {df.shape}')