import pytest
from number_sums_solver.components.square import Square
from number_sums_solver.components.solver import Solver  # Mock this in tests if needed
from number_sums_solver.components.group import Group


@pytest.mark.parametrize(
    "squares, raise_error",
    [
        ([Square(0, 0, 5), Square(1, 0, 10)], False),
        ([Square(0, 0, 5), Square(0, 0, 10)], True), # dup coord
    ]
)
def test_group_unique_coordinates(squares, raise_error):
    if raise_error:
        with pytest.raises(ValueError, match="non unique coordinates in Squares"):
            Group(squares, 15)
    else:
        Group(squares, 15)


@pytest.mark.parametrize(
    "squares, running_target",
    [
        ([Square(0, 0, 5, selected=False), Square(1, 1, 10, selected=True)], 5), 
        ([Square(0, 0, 5, selected=True), Square(1, 1, 10, selected=False)], 10),
        ([Square(0, 0, 5, selected=True), Square(1, 1, 10, selected=True)], 0),
    ]
)
def test_group_running_target(squares, running_target):
    assert Group(squares, 15).running_target == running_target


@pytest.mark.parametrize(
    "squares, available_squares",
    [
        ([Square(0, 0, 5, active=True, selected=False), Square(1, 1, 10, active=False)], [Square(0, 0, 5)]), 
        ([Square(0, 0, 5, active=True, selected=True), Square(1, 1, 10, active=False)], []),
    ]
)
def test_group_available_squares(squares, available_squares):
    assert Group(squares, 15).available_squares == available_squares


@pytest.mark.parametrize(
    "squares, solved",
    [
        ([Square(0, 0, 5, selected=True), Square(1, 1, 10, selected=True)], True), 
        ([Square(0, 0, 5, selected=True), Square(1, 1, 5, selected=True)], False),
    ]
)
def test_group_solved(squares, solved):
    assert Group(squares, 15).solved == solved


@pytest.mark.parametrize(
    "squares, target, selected_tup, deactivated_tup",
    [
        ([Square(0, 0, 5), Square(1, 1, 10)], 5, (True, False), (False, True)),
        ([Square(0, 0, 5), Square(1, 1, 5), Square(0,1,4)], 5, (False, False, False), (False, False, True)),
        ([Square(0, 0, 5), Square(1, 1, 5)], 5, (False, False), (False, False)),
        ([Square(0, 0, 5, selected=True), Square(1, 1, 10, selected=True)], 15, (True, True), (False, False)),
        ([Square(0, 0, 5), Square(1, 1, 5, selected=True), Square(0, 1, 10)], 5, (False, True, False), (True, False, True)),
    ]
)
def test_group_squeeze(squares, target, selected_tup, deactivated_tup):
    group = Group(squares, target)
    group.squeeze(target)
    for square, selected in zip(group.squares, selected_tup):
        assert square.selected == selected
    for square, deactive in zip(group.squares, deactivated_tup):
        assert square.active != deactive