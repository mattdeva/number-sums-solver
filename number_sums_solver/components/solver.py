import itertools
from typing import Sequence

from number_sums_solver.components.utils import _flatten_and_unique

class Solver:
    def __init__(self, integers:list[int]):
        self.integers = integers

    @staticmethod
    def _get_sums_from_sequences(sequences:Sequence[Sequence[int]]):
        ''' consistent way to create a list of sums from list, tuples, generators '''
        return [sum(i) for i in sequences]
    
    def __len__(self):
        return len(self.integers)

    @property
    def addens_sum_list(self) -> dict[Sequence, int]: # being created each time not great but should be fine...
        out_list = []
        for i in range(1, len(self)+1): # loop through number of integers
            combinations = itertools.combinations(self.integers, i) # create combinations of i
            out_list.extend(combinations) # add sequence(s) of integers to list

        return list(zip(
            out_list, # sequences of integers
            self._get_sums_from_sequences(out_list) # sum of integers
        ))
    
    def squeeze(self, sum_:int) -> list[tuple]:
        ''' filter dictionary of {Sequence:int} to only include items wehre sum(Sequence) = int '''
        return [k for k,v in self.addens_sum_list if v==sum_]
    
    def get_addens(self, sum_:int) -> list[int]:
        return _flatten_and_unique(self.squeeze(sum_))
    
# Solver([1,2,2,3,5,6]).squeeze(4)
# Solver([1,1,1,1]).squeeze(4)
# Solver([2]).squeeze(2)
# Solver([1,1,1]).squeeze(2)

# square
