import pytest

from number_sums_solver.components.square import Square

class TestSquare:
    def test_initialization(self):
        square = Square(1, 2, 3, "red", True, False)
        assert square.r == 1
        assert square.c == 2
        assert square.value == 3
        assert square.color == "red"
        assert square.active is True
        assert square.selected is False

    def test_equals(self):
        square1 = Square(1, 2, 3)
        square2 = Square(1, 2, 3)
        square3 = Square(1, 2, 4)  # different value
        assert square1 == square2 
        assert square1 != square3
        assert square1 != "NotASquare"  # non-Square objects 

    def test_hash(self):
        square = Square(1, 2, 3)
        assert hash(square) == hash(Square(1, 2, 3)) 
        assert hash(square) != hash(Square(1, 2, 4))  # different hashes

    def test_coordinates(self):
        assert Square(4, 5, 6).coordinates == (4, 5)  # (row, col)

    # Test default attribute values
    def test_default_values(self):
        square = Square(1, 2, 3)
        assert square.color is None 
        assert square.active is True
        assert square.selected is False

    def test_select(self):
        square = Square(1, 2, 3) # default not selected (see above)
        square.select()
        assert square.selected is True

    # Test the deactivate method
    def test_deactivate(self):
        square = Square(1, 2, 3) # default active (see above)
        square.deactivate()
        assert square.active is False
