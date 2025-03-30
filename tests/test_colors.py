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
    colors = Colors.from_excel('data/dummy_puzzles.xlsx')
    pd.testing.assert_frame_equal(col_df, colors.col_df)
    assert {'FFE06666':6, 'FF6FA8DC':6} == colors.target_dict
    assert sorted(['FFE06666','FF6FA8DC']) == sorted(colors.values)
