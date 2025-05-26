import pytest

from number_sums_solver.images.matrix_image import MatrixImage
from number_sums_solver.images.region import _int_from_img

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
    assert _int_from_img(matrix_image.grey[y:y + h, x:x + w]) == output