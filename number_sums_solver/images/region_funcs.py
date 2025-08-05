# not a yuge fan of script just for functions. but also not sure where else to put these tho...
import numpy as np
import geopandas as gpd
from shapely import Polygon
from functools import reduce

from number_sums_solver.images.region import Region

def gdf_from_regions(regions:list[Region]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({
        'region':regions,
        'polygon':[r.polygon for r in regions],
        'geometry':[r.polygon for r in regions]
    })

def get_neighboring_regions(regions:list[Region]) -> gpd.GeoDataFrame:
    gdf = gdf_from_regions(regions) # create gdf
    gdf = gdf.sjoin(gdf, predicate="intersects", how="left") # join on itself
    gdf['index_left'] = gdf.index # create col from index (probably another way to do this)
    gdf = gdf[gdf.apply(lambda x: x['region_left']._x < x['region_right']._x, axis=1)] # return left/right respective pairs
    return gdf

def avg_polygon_intersection(polygon1:Polygon, polygon2:Polygon):
    return round((
        (polygon1.intersection(polygon2).area / polygon1.area ) + (polygon2.intersection(polygon1).area / polygon2.area)
    ) / 2 * 100, 2)

# NOTE: reducing overlap threshold bc i reducing the Region default areas
def filter_intersection_area(gdf:gpd.GeoDataFrame, overlap_threshold:int=10, left_column:str='polygon_left', right_column:str='polygon_right'):
    gdf['intersection_area'] = gdf.apply(lambda x: avg_polygon_intersection(x[left_column], x[right_column]), axis=1)
    gdf = gdf[gdf.intersection_area > overlap_threshold]
    gdf = gdf.drop(['intersection_area'], axis=1)
    return gdf

def merge_regions(gdf:gpd.GeoDataFrame, left_column:str='region_left', right_column:str='region_right') -> tuple[list[Region]]:
    gdf_ = gdf.copy()

    merge_regions = []
    drop_regions = []

    i = 0
    while len(gdf_) > 0 and i < 10:
        region = list(gdf_[left_column])[0] # get first region on left
        
        neighboring_regions = list(gdf[gdf[left_column] == region][right_column]) # get all neighboring regions

        regions = [region] + neighboring_regions # list of regions to merge then drop

        merge_regions.append(
            reduce(lambda r1, r2: r1.merge(r2), regions) # merge list of regions
        )

        # drop regions
        drop_regions.extend(regions) 
        gdf_ = gdf_[(~gdf_[left_column].isin(drop_regions)) & (~gdf_[right_column].isin(drop_regions))]

        i+=1

    if len(gdf_) > 0:
        ValueError('couldnt merge all regions')
    else:
        return merge_regions, drop_regions

def recursive_split(data, return_indicies:bool=True, max_clusters=4, min_size=2):
    ''' split 1D data into clusters. copilot function '''
    data = np.array(data)
    sorted_indices = np.argsort(data)
    sorted_data = data[sorted_indices]

    clusters = [np.arange(len(sorted_data))]

    while len(clusters) < max_clusters:
        
        # find cluster with highest variance
        idx_to_split = np.argmax([np.var(sorted_data[c]) for c in clusters])
        cluster = clusters[idx_to_split]

        if len(cluster) < min_size * 2:
            break

        best_split, best_var = None, float('inf')

        # loop through each possible split at i, index
        for i in range(min_size, len(cluster) - min_size):
            c1, c2 = cluster[:i], cluster[i:] # split at index
            total_var = np.var(sorted_data[c1]) + np.var(sorted_data[c2]) # calc variance
            if total_var < best_var: # if variance improved (decreases, update variables)
                best_split = (c1, c2)
                best_var = total_var

        if best_split:
            clusters[idx_to_split] = best_split[0]
            clusters.insert(idx_to_split + 1, best_split[1])
        else:
            break

    # return the clusters or the indices
    if return_indicies:
        return [sorted_indices[cluster].tolist() for cluster in clusters]
    else:
        return [sorted_data[cluster].tolist() for cluster in clusters]
    