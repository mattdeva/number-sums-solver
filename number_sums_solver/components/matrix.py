import pandas as pd
from itertools import product

from number_sums_solver.components.square import Square
from number_sums_solver.components.group import Group

def _is_square_df(df:pd.DataFrame):
    if df.shape[0] != df.shape[1]:
        raise ValueError(f'DataFrame shape must be square. got {df.shape}')

def _get_square_coords(df:pd.DataFrame) -> list[tuple[int]]:
    return [t for t in list(product(range(len(df)), repeat=2)) if 0 not in t]

class Matrix:
    def __init__(self, num_df:pd.DataFrame, col_df:pd.DataFrame|None=None):
        
        ## checks to see if df is square
        _is_square_df(num_df)
        if col_df:
            _is_square_df(col_df)
        
        ## init
        self.num_df = num_df
        self.col_df = col_df
        self.squares = self._make_squares(self.num_df, self.col_df)
        self.groups_dict = self._get_groups_dict()
    
    def __len__(self):
        return int(len(self.squares)**(1/2))
    
    def __getitem__(self, input_):
        # prob not best practice to have multiple options.. is fine :)
        if isinstance(input_, str):
            if input_ in self.colors:
                ...
            else:
                return self.groups_dict[input_]
        elif isinstance(self, tuple):
            return self._get_square(input_[0], input_[1])
    
    @property
    def colors(self) -> list[str]:
        return list(set([s.color for s in self.squares if s.color is not None]))
    
    @property
    def active_squares(self) -> list[Square]:
        return [s for s in self.squares if s.active]
    
    @property
    def deactivated_squares(self) -> list[Square]:
        return [s for s in self.squares if not s.active]
    
    @property
    def selected_squares(self) -> list[Square]:
        return [s for s in self.squares if s.selected]
    
    @property
    def groups(self) -> list[Group]:
        return list(self.groups_dict.values())
    
    @property
    def unsolved_groups(self) -> list[Group]:
        return [g for g in self.groups if not g.solved]
    
    @property
    def solved(self) -> bool:
        return all([True if g.solved else False for g in self.groups])

    @staticmethod
    def _make_squares(num_df:pd.DataFrame, col_df:pd.DataFrame|None=None) -> list[Square]:
        coords = _get_square_coords(num_df)
        squares = []
        for coord in coords:
            r,c = coord[0], coord[1]
            value = num_df.iloc[r, c]
            square = Square(r, c, value)
            if col_df:
                square.color = col_df.iloc[r,c]
            squares.append(Square(r, c, value))
        return squares
    
    def _get_groups_dict(self) -> dict[str, Group]:
        groups_dict = {}

        # cols
        for col_i in range(1, len(self)+1):
            groups_dict[f'C{col_i}'] = Group(self._get_square_list(col_i, 1), self._get_target_value(col_i, 1))

        # rows
        for row_i in range(1, len(self)+1):
            groups_dict[f'R{row_i}'] = Group(self._get_square_list(row_i, 0), self._get_target_value(row_i,0))

        # colors
        # create a _get_square_colors #TODO

        return groups_dict

    def _get_square(self, r:int, c:int) -> Square:
        return [s for s in self.squares if s.r==r and s.c==c][0] 
    
    def _get_square_list(self, i:int, axis:int) -> list[Square]:
        if i not in range(1,len(self)+1): # Make Error
            raise ValueError(f'target not in range [1,{len(self)}]')
        
        if axis not in [0,1]: # Make Error
            raise ValueError(f'axis must be in [0,1]. got {axis}')
        
        if axis==0:
            return [self._get_square(r,c) for r,c in zip([i]*len(self), range(1,len(self)+1))]
        if axis==1:
            return [self._get_square(r,c) for r,c in zip(range(1,len(self)+1),[i]*len(self))]

    def _get_target_value(self, i:int, axis:int) -> int:
        if i not in range(1,len(self)+1): # Make Error
            raise ValueError(f'target not in range [1,{len(self)}]')
        
        if axis not in [0,1]: # Make Error
            raise ValueError(f'axis must be in [0,1]. got {axis}')
        
        if axis==0:
            return self.num_df.iloc[i][0].item()
        else:
            return self.num_df.iloc[:,i][0].item()
        
    def get_group_series(self, s:str) -> Group:
        return self.groups_dict[s]
        
    def solve(self, max_iterations=5):

        for _ in range(max_iterations): # sloppy.. should do while incomplete or loop with improvement
            for group in self.unsolved_groups:
                group.squeeze()

            if self.solved:
                print('Solved!')
                return
        print('cant solve :(')