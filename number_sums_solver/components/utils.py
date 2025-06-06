import pandas as pd
import math

from typing import Sequence


def _flatten_and_unique(sequences:Sequence[Sequence[int]]):
    ''' get unique values from a sequence of sequences '''
    # NOTE: will not work on alrady flat lists (should consider making a function to detect this?)
    return sorted(set(i for sequence in sequences for i in sequence))

def _is_square_df(df:pd.DataFrame):
    if df.shape[0] != df.shape[1]:
        raise ValueError(f'DataFrame shape must be square. got {df.shape}')
    
def len_is_sqrt(sequence:int) -> bool:
    length = len(sequence)
    return math.isqrt(length)**2 == length
    
def df_from_value_list(sequence:Sequence[int]) -> pd.DataFrame:
    if not len_is_sqrt(sequence):
        raise ValueError("length of the list must be a perfect square. got {length}")
    
    n = math.isqrt(len(sequence)) 
    
    # Reshape the list row-wise into a matrix
    return pd.DataFrame([sequence[i * n:(i + 1) * n] for i in range(n)]) # create records of length n from list