import pytest
import pandas as pd

from number_sums_solver.images.region import Region
from number_sums_solver.images.matrix_image import (
    MatrixImage,
    _cluster_points,
    _sort_regions,
    _df_from_value_list,
)

@pytest.fixture
def matrix_image():
    return MatrixImage.from_path(
        'data/img_puzzle1.jpeg',
    )

def test_cluster_points(matrix_image):
    not_clustered_regions = [
        Region(matrix_image.image, 100, 155, 10, 10),
        Region(matrix_image.image, 200, 245, 10, 10),
        Region(matrix_image.image, 200, 150, 10, 10),
        Region(matrix_image.image, 100, 250, 10, 10),
    ]
    clustered_regions = [
        [
            Region(matrix_image.image, 200, 150, 10, 10),
            Region(matrix_image.image, 100, 155, 10, 10),
        ],
        [
            Region(matrix_image.image, 200, 245, 10, 10),
            Region(matrix_image.image, 100, 250, 10, 10),
        ]
    ]
    assert _cluster_points(not_clustered_regions) == clustered_regions

def test_align_regions(matrix_image):
    not_sorted_region_list = [
        Region(matrix_image.image, 100, 255, 10, 10),
        Region(matrix_image.image, 300, 245, 10, 10),
        Region(matrix_image.image, 200, 250, 10, 10),
    ]
    sorted_region_list = [
        Region(matrix_image.image, 100, 255, 10, 10),
        Region(matrix_image.image, 200, 250, 10, 10),
        Region(matrix_image.image, 300, 245, 10, 10),
    ]
    assert _sort_regions(not_sorted_region_list) == sorted_region_list

# NOTE: eventually update this to run when prompted
@pytest.mark.skip(reason="takes 10 seconds...")
def test_df_from_matrix_image(matrix_image):
    pd.testing.assert_frame_equal(
        matrix_image.to_matrix().num_df, 
        pd.DataFrame({
            0:[0,19,9,11,10],
            1:[6,2,8,4,2],
            2:[3,4,7,8,3],
            3:[21,8,6,1,7],
            4:[19,9,3,7,7]
        })
    )
