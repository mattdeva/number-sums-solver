import pytest

from number_sums_solver.images.tile import (
    _get_len,
    _get_size,
    _get_random_color,
    _get_color
)

@pytest.mark.parametrize("list_, output, error", [
    ([1, 2, 3], '', True),
    ([1, 2, 3, 4, 5, 6, 7, 8, 9], 4, False), 
])
def test_get_len(list_, output, error):
    if error:
        with pytest.raises(ValueError):
            _get_len(list_)
    else:
        assert _get_len(list_) == output

@pytest.mark.parametrize("input_, output, error", [
    (5, (5, 5, 3), False),
    ((10, 10, 3), (10, 10, 3), False),
    ((10, 10), '', True),
])
def test_get_size(input_, output, error):
    if error:
        with pytest.raises(ValueError):
            _get_size(input_)
    else:
        assert _get_size(input_) == output

@pytest.mark.skip(reason="may not need in current form or at all.")
@pytest.mark.parametrize("seed, len_", [
    (None, 2),
    (42, 2),
])
def test_get_random_color(seed, len_):
    tup = _get_random_color(seed)
    assert len(tup) == len_
    assert isinstance(tup, tuple)
    assert isinstance(tup[0], str)
    assert isinstance(tup[1], tuple)
    assert len(tup[1]) == 3

@pytest.mark.parametrize("input_, output", [
    ((255, 0, 0), (255, 0, 0)), 
    ("red", (255, 0, 0)), 
    # NOTE: doesnt have the random color output from invalid input bc that part doesnt work yet.
])
def test_get_color(input_, output):
    if isinstance(output, type) and issubclass(output, Exception):
        with output:
            _get_color(input_)
    else:
        assert _get_color(input_) == output

