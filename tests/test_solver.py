import pytest

from number_sums_solver.components.solver import Solver

@pytest.fixture
def solver1():
    return Solver([1,2,2,3,5,6])

@pytest.fixture
def solver2():
    return Solver([1,1,1,1])

@pytest.fixture
def solver3():
    return Solver([2])

@pytest.fixture
def solver4():
    return Solver([1,1,1])

def test_addens_sum_list(solver3, solver4):
    assert solver4.addens_sum_list == ([((1,), 1),((1,), 1),((1,), 1),((1, 1), 2),((1, 1), 2),((1, 1), 2),((1, 1, 1), 3)])
    assert solver3.addens_sum_list == ([((2,), 2)])

def test_squeeze(solver1, solver2, solver3, solver4):
    assert solver1.squeeze(4) == [(1, 3), (2, 2)]
    assert solver2.squeeze(4) == [(1, 1, 1, 1)]
    assert solver3.squeeze(2) == [(2,)]
    assert solver4.squeeze(2) == [(1, 1), (1, 1), (1, 1)]

def test_get_addens(solver1, solver2, solver3, solver4):
    assert solver1.get_addens(4) == [1,2,3]
    assert solver2.get_addens(4) == [1]
    assert solver3.get_addens(2) == [2]
    assert solver4.get_addens(2) == [1]