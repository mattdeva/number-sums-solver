import pandas as pd
from typing import Sequence

def _flatten_and_unique(sequences:Sequence[Sequence[int]]):
    ''' get unique values from a sequence of sequences '''
    # NOTE: will not work on alrady flat lists (should consider making a function to detect this?)
    return sorted(set(i for sequence in sequences for i in sequence))

def _is_square_df(df:pd.DataFrame):
    if df.shape[0] != df.shape[1]:
        raise ValueError(f'DataFrame shape must be square. got {df.shape}')