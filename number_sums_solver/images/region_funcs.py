# not a yuge fan of script just for functions. but also not sure where else to put these tho...

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

def filter_intersection_area(gdf:gpd.GeoDataFrame, overlap_threshold:int=50, left_column:str='polygon_left', right_column:str='polygon_right'):
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
