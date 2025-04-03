import pandas as pd
import numpy as np
from itertools import product

from number_sums_solver.components.square import Square
from number_sums_solver.components.group import Group
from number_sums_solver.components.utils import _is_square_df
from number_sums_solver.components.colors import Colors


def _df_shapes_same(df1:pd.DataFrame, df2:pd.DataFrame):
    if not df1.shape == df2.shape:
        raise ValueError(f'Dataframe shapes do not match')

def _get_square_coords(df:pd.DataFrame) -> list[tuple[int]]:
    return [t for t in list(product(range(len(df)), repeat=2)) if 0 not in t]

def _pull_cell_value(input_:int|str):
    if isinstance(input_, int):
        return input_
    else:
        return int(input_.split('(')[0].strip())
    
def _pull_cell_values_from_df(input_df:pd.DataFrame): # great name
    df = input_df.copy()
    for col in df:
        df[col] = [_pull_cell_value(i) for i in df[col]]
    return df

def _get_int(input_:object):
    ''' want to return native int, not numpy int.. theres probably a better way to do this '''
    if isinstance(input_, int):
        return input_
    elif hasattr(input_, 'item'):
        return input_.item()
    else:
        raise ValueError(f'Input must be int-like. got {type(input_)}')
class Matrix:
    def __init__(self, num_df:pd.DataFrame, colors:Colors|None=None):
        
        ## checks to see if df is square
        _is_square_df(num_df)
        
        ## init
        # NOTE: i couldve done a 3D array instead of multiple dfs.. but i opted for this.. 
            # i think makes a little easier/straightforward in the case of no colors. but up for debate
        self.num_df = num_df
        self.colors = self._check_colors(self.num_df, colors)

        self.squares = self._make_squares(self.num_df, self.colors)
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
          
    @classmethod
    def from_excel(cls, path:str, sheet_name:str|None=None):
        return cls(
            _pull_cell_values_from_df(pd.read_excel(path, header=None, sheet_name=sheet_name)),
            Colors.from_excel(path)
        )
    
    @property
    def color_values(self) -> list[str]:
        if self.colors is None:
            return []
        else:
            return self.colors.values
    
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
    def _check_colors(num_df:pd.DataFrame, colors:Colors):
        if colors is None: # im sure this is not best practice
            return None 
        
        _df_shapes_same(num_df, colors.col_df)

        return colors

    @staticmethod
    def _make_squares(num_df:pd.DataFrame, colors:Colors|None=None) -> list[Square]:
        coords = _get_square_coords(num_df)
        squares = []
        for coord in coords:
            r,c = coord[0], coord[1]
            value = num_df.iloc[r, c]
            square = Square(r, c, value)
            if colors is not None:
                square.color = colors.col_df.iloc[r,c]
            squares.append(square)
        return squares
    
    def _get_groups_dict(self) -> dict[str, Group]:
        groups_dict = {}

        # cols
        for col_i in range(1, len(self)+1):
            groups_dict[f'C{col_i}'] = Group(
                self._get_square_list(col_i, 1), 
                self._get_target_value(col_i, 1)
            )

        # rows
        for row_i in range(1, len(self)+1):
            groups_dict[f'R{row_i}'] = Group(
                self._get_square_list(row_i, 0), 
                self._get_target_value(row_i,0)
            )

        # colors
        if self.colors is not None:
            for color_str in self.color_values: # NOTE: couldve just done colors.values directly.. probably better practice that way
                groups_dict[color_str] = Group(
                    self._get_square_color(color_str),
                    self.colors.target_dict[color_str]
                )
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
        
    def _get_square_color(self, color_str:str) -> list[Square]:
        return [s for s in self.squares if s.color==color_str]

    def _get_target_value(self, i:int, axis:int) -> int:
        if i not in range(1,len(self)+1): # Make Error
            raise ValueError(f'target not in range [1,{len(self)}]')
        
        if axis not in [0,1]: # Make Error
            raise ValueError(f'axis must be in [0,1]. got {axis}')
        
        if axis==0:
            return _get_int(self.num_df.iloc[i][0])
        else:
            return _get_int(self.num_df.iloc[:,i][0])
        
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