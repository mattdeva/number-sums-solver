import pytest
import pandas as pd

from number_sums_solver.components.utils import _flatten_and_unique, _is_square_df

@pytest.mark.parametrize("input_, output", [
    ([[1,2,3]], [1,2,3]),  # one flat list
    ([[1,2], [3]], [1,2,3]),  # nested list
    ([[1,2], [2,3]], [1,2,3]),  #  unique values and order
])
def test_flatten_and_unique(input_, output):
    assert _flatten_and_unique(input_) == output

@pytest.mark.parametrize("df, raise_error", [
    (pd.DataFrame([[1,2], [3,4]]), False),  # square
    (pd.DataFrame([[1,2,3], [4,5,6]]), True),  # not square
])
def test_is_square_df(df, raise_error):
    if raise_error:
        with pytest.raises(ValueError, match="DataFrame shape must be square"):
            _is_square_df(df)
    else:
        _is_square_df(df)  # Should not raise an error