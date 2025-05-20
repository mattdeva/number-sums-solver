import pytest
import pandas as pd
import numpy as np

from number_sums_solver.components.square import Square
from number_sums_solver.components.group import Group
from number_sums_solver.components.colors import Colors
from number_sums_solver.components.matrix import (
    _df_shapes_same,
    _get_square_coords,
    _pull_cell_value,
    _pull_cell_values_from_df,
    _get_int,
    Matrix
)

@pytest.fixture
def matrix_1():
    return Matrix.from_excel(
        'data/dummy_puzzles.xlsx',
        'plain_ex1'
    )

@pytest.fixture
def matrix_2():
    return Matrix.from_excel(
        'data/dummy_puzzles.xlsx',
        'colors_ex1'
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


def test_solve(matrix_1): 
    ## also tests
        # selected squares
        # deactivated squares
        # unsolved_groups

    m = matrix_1
    assert m.deactivated_squares == []
    assert m.selected_squares == []
    assert len(m.unsolved_groups) == 6
    assert m.solved == False
    m.solve()
    assert m.selected_squares == [Square(1,1,3), Square(1,2,1), Square(2,1,2), Square(2,3,4), Square(3,2,2)]
    assert m.active_squares == [Square(1,1,3), Square(1,2,1), Square(2,1,2), Square(2,3,4), Square(3,2,2)]
    assert m.deactivated_squares == [Square(1,3,4,False,False), Square(2,2,3,False,False), Square(3,1,2,False,False), Square(3,3,1,False,False)]
    assert m.unsolved_groups == []
    assert m.solved == True

def test_color_values(matrix_1, matrix_2):
    assert matrix_1.color_values == []
    print(len(matrix_2))
    assert sorted(matrix_2.color_values) == ['6FA8DC', 'E06666']

def test_groups(matrix_1, matrix_2):
    assert len(matrix_1.groups) == 6
    assert len(matrix_2.groups) == 8

def test_get_square(matrix_1):
    assert matrix_1._get_square(1,1) == Square(1,1,3)

@pytest.mark.parametrize("i, axis, squares, raise_error", [
    (1,0,[Square(1,1,3), Square(1,2,1), Square(1,3,4)],False), 
    (1,1,[Square(1,1,3), Square(2,1,2), Square(3,1,2)],False),
    (0,0,[],True),
    (4,0,[],True),
    (1,2,[],True),
])
def test_get_square_list(i, axis, squares, raise_error, matrix_1):
    if raise_error:
        with pytest.raises(ValueError):
            matrix_1._get_square_list(i,axis)
    else:
        assert matrix_1._get_square_list(i,axis) == squares

def test_get_square_color(matrix_2):
    
    matrix_2._get_square_color('FF6FA8DC') == [Square(1,3,3), Square(2,3,1), Square(3,1,2), Square(3,2,2), Square(3,3,3)]
    matrix_2._get_square_color('FFE06666') == [Square(1,1,3), Square(1,2,1), Square(2,1,2), Square(2,2,2)]

@pytest.mark.parametrize("i, axis, target_value, raise_error", [
    (1,0,4,False), 
    (1,1,5,False),
    (0,0,'',True),
    (4,0,'',True),
    (1,2,'',True),
])
def test_get_target_value(i, axis, target_value, raise_error, matrix_1):
    if raise_error:
        with pytest.raises(ValueError):
            matrix_1._get_target_value(i,axis)
    else:
        assert matrix_1._get_target_value(i,axis) == target_value

@pytest.mark.parametrize("tuple, output_type",[
    ((1,1), Square),
    ((0,1), Group),
    ((1,0), Group),
])
def test_get_tile(tuple, output_type, matrix_1):
    assert isinstance(matrix_1.get_tile(tuple), output_type)

def test_zeros():
    zeros_w_color = Matrix.zeros(2,True)
    zeros_no_color = Matrix.zeros(2,False)

    assert zeros_no_color.colors is None
    assert isinstance(zeros_w_color.colors, Colors)
    pd.testing.assert_frame_equal(
        zeros_no_color.num_df,
        pd.DataFrame(np.zeros((3,3), dtype=int))
    )
    pd.testing.assert_frame_equal(
        zeros_w_color.num_df,
        pd.DataFrame(np.zeros((3,3), dtype=int))
    )

def test_update_value():
    m = Matrix.zeros(3, True)
    m.update_square((0,1), 3)
    m.update_square((1,0), 4)
    m.update_square((1,1), 3, 'red')

    assert isinstance(m.get_tile((0,1)), Group)
    assert m.get_tile((0,1)).nominal_target == 3
    assert isinstance(m.get_tile((1,0)), Group)
    assert m.get_tile((1,0)).nominal_target == 4
    assert isinstance(m.get_tile((1,1)), Square)
    assert m.get_tile((1,1)).value == 3
    assert m.get_tile((1,1)).color == 'red'

def test_update_color():
    m = Matrix.zeros(3, True)
    
    m.update_color_target('red', 6)
    m.update_color_target('blue', 6)
    m.remove_color_taget('white')

    assert m.colors.target_dict == {'red': 6, 'blue': 6}
