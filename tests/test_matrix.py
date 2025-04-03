import pytest
import pandas as pd
import numpy as np

from number_sums_solver.components.matrix import (
    _df_shapes_same,
    _get_square_coords,
    _pull_cell_value,
    _pull_cell_values_from_df,
    _get_int
)

@pytest.mark.parametrize(
    "df1, df2, raise_error",
    [
        (pd.DataFrame([[1, 2], [3, 4]]), pd.DataFrame([[5, 6], [7, 8]]), False), 
        (pd.DataFrame([[1, 2]]), pd.DataFrame([[3, 4], [5, 6]]), True),
    ]
)
def test_df_shapes_same(df1, df2, raise_error):
    if raise_error:
        with pytest.raises(ValueError):
            _df_shapes_same(df1, df2)
    else:
        _df_shapes_same(df1, df2)

@pytest.mark.parametrize(
    "df, coords",
    [
        (pd.DataFrame([[1, 2], [3, 4]]), [(1, 1)]),
        (pd.DataFrame([[0, 1], [1, 2]]), [(1, 1)]), # ignores coords with 0
    ]
)
def test_get_square_coords(df, coords):
    assert _get_square_coords(df) == coords

# Tests for _pull_cell_value
@pytest.mark.parametrize(
    "input_, int_",
    [
        (1, 1),
        ("3(9)", 3),
        ("7 (24)", 7),
    ]
)
def test_pull_cell_value(input_, int_):
    assert _pull_cell_value(input_) == int_

# Tests for _pull_cell_values_from_df
@pytest.mark.parametrize(
    "input_df, output_df",
    [
        (pd.DataFrame([["3", "6(10)"], ["7 (15)", "9"]]),
         pd.DataFrame([[3, 6], [7, 9]])),  # Conversion of strings to integers
    ]
)
def test_pull_cell_values_from_df(input_df, output_df):
    pd.testing.assert_frame_equal(_pull_cell_values_from_df(input_df), output_df)

# Tests for _get_int
@pytest.mark.parametrize(
    "input_, expected_output, raise_error",
    [
        (5, 5, False),
        (np.int32(5), 5, False),
        ("string", None, True),
    ]
)
def test_get_int(input_, expected_output, raise_error):
    if raise_error:
        with pytest.raises(ValueError):
            _get_int(input_)
    else:
        assert _get_int(input_) == expected_output