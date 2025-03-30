import pandas as pd
import numpy as np
from openpyxl import load_workbook

from number_sums_solver.components.utils import _is_square_df

    
def _df_unique_values(df:pd.DataFrame) -> list:
    return [v for v in np.unique(df.values) if v != df.iloc[0,0]]

def _get_colors_components(excel_path:str) -> tuple[pd.DataFrame, dict]:
    # assumes one sheet in excel for now (planning to change later)
    # i dont think theres a point to break up function even tho it does a couple things

    wb = load_workbook(excel_path)
    sheet = wb.active

    rows = []
    dict_ = {}
    for row in sheet.iter_rows():
        record = []
        for cell in row:
            if cell.fill and cell.fill.fgColor:
                record.append(cell.fill.fgColor.rgb)
            else:
                record.append(None)

            if isinstance(cell.value, tuple([float, int])):
                continue

            if '(' in cell.value:
                dict_[cell.fill.fgColor.rgb] = int(cell.value.split('(')[-1].split(')')[0])
            
        rows.append(record)

    return pd.DataFrame(rows), dict_

class Colors:
    def __init__(self, col_df:pd.DataFrame, target_dict:dict[str, int]):
        self.col_df = col_df
        self.target_dict = target_dict
        self._check_input()

    def _check_input(self):
        _is_square_df(self.col_df)

        df_values = _df_unique_values(self.col_df)
        dict_keys = list(self.target_dict)

        if not sorted(df_values) == sorted(dict_keys):
            raise ValueError(f'DataFrame and Target_Dict must have same values. got {df_values} and {dict_keys}')
        
    @classmethod
    def from_excel(cls, path:str):
        col_df, target_dict = _get_colors_components(path)
        return cls(col_df, target_dict) 

    @property
    def values(self):
        return list(self.target_dict)
        