import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict

from number_sums_solver.components.colors import Colors
from number_sums_solver.components.matrix import Matrix
from number_sums_solver.components.utils import len_is_sqrt, df_from_value_list
from number_sums_solver.components.square import Square
from number_sums_solver.images.region import Region
from number_sums_solver.images.region_funcs import (
    get_neighboring_regions,
    filter_intersection_area,
    merge_regions,
    recursive_split
)

from typing import Sequence

# also TODO: limit available colors so that we dont have to worry (as much) about stuff like this :/ (also this is in region...)
WHITES = ['white', 'whitesmoke', 'gainsboro', 'linen', 'mintcream', 'ivory', 'ghostwhite']


def _cluster_points(regions:list[Region], epsilon:int=25) -> list[list[Region]]:
    ''' copilot code -- clusters coordinates into groups based on y values '''
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

def _detect_colored_background(image, s_thresh=50, v_thresh=100):
    ''' copilot function. find non white and non black pixels from images '''    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    brightness = hsv[:, :, 2]

    # Colorful pixels (likely tiles) — not white, not black
    mask = cv2.inRange(saturation, s_thresh, 255) & cv2.inRange(brightness, v_thresh, 255)
    return mask

def _wipe_colored_tiles(image):
    ''' copilot function. replace non white and non black pixels with white pixels '''
    mask = _detect_colored_background(image)
    cleaned = image.copy()

    # Only update pixels identified as tile backgrounds
    cleaned[mask > 0] = (255, 255, 255)
    return cleaned

def _strengthen_edges(edge_img, kernel_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.dilate(edge_img, kernel, iterations=1)

def _euclidean_distance(c1:tuple[int], c2:tuple[int]) -> float:
    return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5

def _find_closest_coordinate_adj(coordinates:tuple[int], coordinates_options:list[tuple[int]], return_index:bool=True) -> list[int|tuple[int]]:
    indexed_options = list(enumerate(coordinates_options))
    filtered_options = [(i,c) for i,c in indexed_options if c[0] > coordinates[0] and c[1] > coordinates[1]]

    closest_index, closest_coord = min(filtered_options, key=lambda pair: _euclidean_distance(coordinates, pair[1]))

    return closest_index if return_index else closest_coord


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
    def whiteout(self) -> np.ndarray:
        return _wipe_colored_tiles(self.image)
    
    @property
    def grey(self) -> np.ndarray:
        return cv2.cvtColor(self.whiteout, self._gray_code)
    
    @property
    def edge(self) -> np.ndarray:
        return _strengthen_edges(
            cv2.Canny(self.grey, self._edge_upper_threshold, self._edge_lower_threshold)
        )
    
    @property
    def contours(self) -> tuple[np.ndarray]:
        return cv2.findContours(self.edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    @property
    def tile_regions(self) -> list[Region]:
        return [Region.from_contour(self.image, c) for c in self.contours]

    @property
    def processed_regions(self) -> list[Region]:
        if self._processed_regions is None:
            print('processed regions only available after calling `to_matrix`. this is meant for dev only.')
        return self._processed_regions
    
    @property
    def all_colors(self) -> list[str]:
        return [r.color for r in self.tile_regions]
    
    @property
    def non_white_colors(self) -> list[str]:
        return [c for c in self.all_colors if c not in WHITES]
    
    @property
    def puzzle_has_colors(self) -> bool:
        return True if len(self.non_white_colors) > 2 else False
    
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

    def get_region_clusters(self, *args, **kwargs):
        return recursive_split([r.area for r in self.tile_regions], *args, **kwargs)

    def to_matrix(self) -> Matrix:

        # 0. Determine if color puzzle or not and create regions accordingly
        if self.puzzle_has_colors:
            clusters = self.get_region_clusters()
            color_label_indices = clusters[0]
            tile_indicies = list(itertools.chain.from_iterable(clusters[1:]))

            color_label_regions = [self.tile_regions[i] for i in color_label_indices]
            regions = [self.tile_regions[i] for i in tile_indicies]
            # return regions

        else:
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

        # 9. If color puzzle, make Colors, else None
        if self.puzzle_has_colors:
            ## copying the process to find/merge to double digits ## (not great TODO: clean this up)
            real_neighbors_gdf_CL = get_neighboring_regions(color_label_regions) # great variable name 
            merged_dubdigit_regions_CL, drop_dubdigit_regions_CL = merge_regions(real_neighbors_gdf_CL) # great variable name x2
            regions_CL = [r for r in color_label_regions if r not in drop_dubdigit_regions_CL]
            regions_CL.extend(merged_dubdigit_regions_CL)
            region_clusters_CL = _cluster_points(regions_CL) # region_clusters: list[list[Region]]
            region_clusters_CL = [_sort_regions(l) for l in region_clusters_CL]
            aligned_regions_CL = list(itertools.chain.from_iterable(region_clusters_CL))
            ###

            # store the found color lables for debugging help
            self._processed_color_label_regions = aligned_regions_CL

            # find the closest tile regions to the right and down from each color label tile
            tile_region_indexes = [_find_closest_coordinate_adj(c, [r.coordinates for r in aligned_regions]) for c in [r.coordinates for r in aligned_regions_CL]]

            # create col_df
            col_df = df_from_value_list(['white']+[s.color for s in squares])

            # create Colors
            target_dict = {aligned_regions[i].color:r.value for i,r in zip(tile_region_indexes, aligned_regions_CL)}
            self._processed_col_df = col_df
            colors = Colors(col_df, target_dict)
        else:
            colors = None

        return Matrix.from_squares(
            squares,
            colors
        )
