import pytest
import pandas as pd

from number_sums_solver.images.matrix_image import (
    MatrixImage,
    int_from_img,
    find_pairs,
    concat_digits,
    create_multi_digit_dict,
    update_value_dict,
    floor_round,
    df_from_value_list,
    create_value_tuple
)

# specifc tests to dummy data ok for this

@pytest.fixture
def matrix_image():
    return MatrixImage.from_path(
        'data/img_puzzle1.jpeg',
    )


@pytest.mark.parametrize(
    "x, y, h, w, output",
    [
        (5,5,5,5,None), 
        (113,958,75,40,1),
        (351,944,95,65,2)
    ]
)
def test_int_from_img(x, y, h, w, output, matrix_image):
    assert int_from_img(matrix_image.grey[y:y + h, x:x + w]) == output

@pytest.mark.parametrize(
    "coordinates, buffers, pairs, raise_error",
    [
        ([(1,2),(2,3)], (3,3), [((1,2),(2,3))], False),
        ([(1,2),(4,5)], (3,3), [], False),
        ([(1,2),(4,5)], (4,4), [((1, 2), (4, 5))], False),
        ([(10,20),(12,25),(12,22),(16,24),(13,26)], (3,3),[((10, 20), (12, 22)), ((12, 25), (13, 26))], False),
        ([(10,20),(12,25),(12,22),(16,24),(13,26)], (10,10),[((10, 20), (12, 22)), ((12, 25), (13, 26))], True),
    ]
)
def test_find_pairs(coordinates, buffers, pairs, raise_error):
    x_buffer, y_buffer = buffers
    if raise_error:
        with pytest.raises(ValueError):
            find_pairs(coordinates, x_buffer, y_buffer)
    else:
        assert find_pairs(coordinates, x_buffer, y_buffer) == pairs

@pytest.mark.parametrize(
    "arg1, arg2, output",
    [
        (1, 2, 12),
        ("1", "2", 12),
        (1, "2", 12),
    ]
)
def test_concat_digits(arg1, arg2, output):
    assert concat_digits(arg1, arg2) == output

def test_create_multi_digit_dict():
    # one test is fine.. basically just looking up in dictionary and concat_digits (which is already tested)
    assert create_multi_digit_dict(
        [((1,2),(3,4)), ((5,6),(7,8))],
        {(1,2):9, (3,4):8, (5,6):7, (7,8):6}
    ) == {(1, 2): 98, (5, 6): 76}

def test_update_value_dict():
    # single test. shows adding 0 entry, replacing multiple coords with double digit based on pairs.
    assert update_value_dict(
        {(1,2):9, (3,4):8, (5,6):7, (7,8):6},
        [((1,2),(3,4))],
        {(1,2):98}
    ) == {(0, 0): 0, (1, 2): 98, (5, 6): 7, (7, 8): 6}

@pytest.mark.parametrize(
    "input, output",
    [
        (200,200),
        (201,200),
        (999,900),
        (1001,1000),
        (1099,1000),
    ]
)
def test_floor_round(input, output):
    assert floor_round(input) == output

def test_create_value_tuple():
    # one simple test should do here (checks dict -> tuple & ordering)
    assert create_value_tuple({(3,4):4, (2,4):3, (1,2):1, (3,2):2}) == [(1, 2, 1), (2, 4, 3), (3, 4, 4), (3, 2, 2)]

def test_df_from_value_list():
    pd.testing.assert_frame_equal(
        df_from_value_list([1,2,3,4]),
        pd.DataFrame({0:[1,3], 1:[2,4]})
    )

    with pytest.raises(ValueError):
        df_from_value_list([1,2,4])
