import pytest
import pandas as pd

from number_sums_solver.components.colors import _df_unique_values, Colors

@pytest.fixture
def col_df():
    return pd.DataFrame([
    ['00000000', '00000000', '00000000', '00000000'],
    ['00000000', 'FFE06666', 'FFE06666', 'FF6FA8DC'],
    ['00000000', 'FFE06666', 'FFE06666', 'FF6FA8DC'],
    ['00000000', 'FF6FA8DC', 'FF6FA8DC', 'FF6FA8DC'],
])

def test_df_unique_values(col_df):
    assert sorted(['FFE06666','FF6FA8DC']) == _df_unique_values(col_df)


def test_colors(col_df): # simple test fine
    colors = Colors.from_excel('data/dummy_puzzles.xlsx', sheet_name='colors_ex1')
    pd.testing.assert_frame_equal(col_df, colors.col_df)
    assert {'FFE06666':6, 'FF6FA8DC':6} == colors.target_dict
    assert sorted(['FFE06666','FF6FA8DC']) == sorted(colors.values)

def test_change_colors_from_dict():
    colors = Colors.from_excel('data/dummy_puzzles.xlsx', sheet_name='colors_ex1')
    colors.change_colors({'FFE06666':'red', 'FF6FA8DC':'blue'})
    colors.target_dict == {'red':6, 'blue':6}
    assert list(colors.col_df[2][1:]) == ['red', 'red', 'blue']

def test_change_colors_from_sequence():
    colors = Colors.from_excel('data/dummy_puzzles.xlsx', sheet_name='colors_ex1')
    colors.change_colors(('red', 'blue'))
    colors.target_dict == {'red':6, 'blue':6}
    assert list(colors.col_df[2][1:]) == ['blue', 'blue', 'red']
