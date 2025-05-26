import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict

from number_sums_solver.components.matrix import Matrix
from number_sums_solver.components.utils import len_is_sqrt
from number_sums_solver.components.square import Square
from number_sums_solver.images.region import Region
from number_sums_solver.images.region_funcs import (
    get_neighboring_regions,
    filter_intersection_area,
    merge_regions
)

from typing import Sequence


def _cluster_points(regions:list[Region], epsilon:int=25) -> list[list[Region]]:
    # copilot code -- clusters coordinates into groups based on y values
    regions.sort(key=lambda r: r._y) # sort by y
    clusters_dict = defaultdict(list) # {y_cluster:int, list[Region]}

    for region in regions:
        assigned = False
        for cluster_y in clusters_dict.keys(): # loop through found unique clusters
            if abs(cluster_y - region._y) <= epsilon:
                clusters_dict[cluster_y].append(region)
                assigned = True # dont need to create cluster
                break
        if not assigned: # if point doesnt belong to known cluster
            clusters_dict[region._y].append(region) # create cluster with y value
    return [v for v in clusters_dict.values()]

def _sort_regions(regions:list[Region]) -> list[Region]:
    return sorted(regions, key=lambda r: r._x)

def _get_squares_from_regions(regions:Sequence[Region]) -> list[Square]:
    # probably my least favorite function in this whole thing...
    if not len_is_sqrt([0]+regions): # not my fav
        raise ValueError(f'regions must be of length 1 less of a perfect sqaure. got {len(regions)}')

    size = int((len(regions)+1)**(1/2))

    squares = []
    i = 0
    for r in range(size):
        for c in range(size):
            if r == 0 and c == 0:
                continue
            squares.append(
                Square(r,c,regions[i].value, regions[i].color)
            )
            i += 1
    return squares

class MatrixImage:
    def __init__(
            self, 
            image:np.ndarray, 
            gray_code=cv2.COLOR_BGR2GRAY, 
            edge_upper_threshold=1000, 
            edge_lower_threshold=350, 
            retr_external=cv2.RETR_EXTERNAL, 
            chain_approx_simple=cv2.CHAIN_APPROX_SIMPLE
        ):
        self.image = image
        self._gray_code = gray_code
        self._edge_upper_threshold = edge_upper_threshold
        self._edge_lower_threshold = edge_lower_threshold
        self._retr_external = retr_external
        self._chain_approx_simple = chain_approx_simple

        # not great practice but helpful for troubleshooting...
        self._processed_tile_value_regions = None
        self._processed_squares = None

    @classmethod
    def from_path(cls, path:str):
        return cls(cv2.imread(path))

    @property
    def grey(self):
        return cv2.cvtColor(self.image, self._gray_code)
    
    @property
    def edge(self):
        return cv2.Canny(self.grey, self._edge_upper_threshold, self._edge_lower_threshold)
    
    @property
    def contours(self):
        return cv2.findContours(self.edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    @property
    def tile_regions(self):
        return [Region.from_contour(self.image, c) for c in self.contours]

    @property
    def processed_regions(self):
        if self._processed_regions is None:
            print('processed regions only available after calling `to_matrix`. this is meant for dev only.')
        return self._processed_regions
    
    def show_image_with_axis(self):
        plt.imshow(self.image)
        
    def show_image(self):
        plt.imshow(self.image)
        plt.axis(False)
        plt.show()

    def show_grey(self):
        plt.imshow(self.grey)
        plt.axis(False)
        plt.show()

    def show_edge(self):
        plt.imshow(self.edge)
        plt.axis(False)
        plt.show()

    def to_matrix(self) -> Matrix:
        # 0. create regions
        regions = self.tile_regions

        ## 1. Find dupliacte regions

        # 1a. Create gdf of neighbors
        initial_neighbors_gdf = get_neighboring_regions(regions)

        # 1b. Filter by intersection area. Only keep regions with high overlap
        overlap_regions_gdf = filter_intersection_area(initial_neighbors_gdf)

        # 1c. Merge overlap regions to one
        merged_overlap_regions, drop_overlap_regions = merge_regions(overlap_regions_gdf)

        ## 2. Remove duplicates. Merge regions with large overlap.

        # 2a. Remove duplicates
        regions = [r for r in regions if r not in drop_overlap_regions]

        # 2b. Add the new merged regions to regions
        regions.extend(merged_overlap_regions)

        ## 3. Find 2-digit numbers (yes hardcapped at 2 digits. Possible future issue.)
        
        # 3a. Create gdf of neighbors
        real_neighbors_gdf = get_neighboring_regions(regions) # great variable name

        # 3b. Create 2 digit numbers via merge
        merged_dubdigit_regions, drop_dubdigit_regions = merge_regions(real_neighbors_gdf) # great variable name x2

        # 4. Remove individual regions from of 2-digit numbers
        regions = [r for r in regions if r not in drop_dubdigit_regions]

        # 5. Add merged double digit regions to regions
        regions.extend(merged_dubdigit_regions)

        # 6. Align regions Y, then X coordinates

        # 6a. Group regions into clusters based on Y coordinates (rows)
        region_clusters = _cluster_points(regions) # region_clusters: list[list[Region]]

        # 6b. Order by X within each cluster
        region_clusters = [_sort_regions(l) for l in region_clusters]

        # 6c. Flatten list
        aligned_regions = list(itertools.chain.from_iterable(region_clusters))

        # 8. Create squares from matrix
        squares = _get_squares_from_regions(aligned_regions)

        # 7. Store these to help debug what went wrong in image processing
        self._processed_tile_value_regions = aligned_regions
        self._processed_squares = squares

        return Matrix.from_squares(
            squares,
            None
        )
