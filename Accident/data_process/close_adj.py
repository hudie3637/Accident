import pandas as pd
import geopandas as gpd
from shapely.wkt import loads
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np

# 读取CSV文件并创建投影后的 GeoDataFrame
Chicago_data = pd.read_csv('../data/Chicago/raw/CommAreas.csv')
NYC_data = pd.read_csv('data/NYC/taxi_zones.csv')

Chicago_data = Chicago_data.dropna()
Chicago_data = Chicago_data.drop_duplicates()
Chicago_data['geometry'] = Chicago_data['the_geom'].apply(lambda wkt_str: loads(wkt_str))
Chicago_gdf = gpd.GeoDataFrame(Chicago_data, geometry='geometry', crs='EPSG:4326')
Chicago_gdf_projected = Chicago_gdf.to_crs(crs='EPSG:3857')

NYC_data = NYC_data.dropna()
NYC_data = NYC_data.drop_duplicates()
NYC_data['geometry'] = NYC_data['the_geom'].apply(lambda wkt_str: loads(wkt_str))
NYC_gdf = gpd.GeoDataFrame(NYC_data, geometry='geometry', crs='EPSG:4326')
NYC_gdf_projected = NYC_gdf.to_crs(crs='EPSG:3857')
# 定义计算距离的函数
def calculate_distance(geom1, geom2):
    return geom1.distance(geom2)
# 创建紧密度邻接矩阵的函数
def create_Chicago_closeness_adjacency_matrix(gdf, beta, e):
    areas = gdf['AREA_NUMBE'].tolist()
    matrix = pd.DataFrame(0, index=areas, columns=areas, dtype=float)

    for i, geom_i in enumerate(gdf['geometry']):
        for j, geom_j in enumerate(gdf['geometry']):
            if i != j:
                distance = calculate_distance(geom_i, geom_j)

                value = np.exp(-beta * (distance ** 2))
                matrix.iloc[i, j] = value if value >= e else 0

    return matrix
# 创建连接图邻接矩阵的函数
def create_Chicago_road_adjacency_matrix(gdf):
    areas = gdf['AREA_NUMBE'].tolist()
    matrix = pd.DataFrame(0, index=areas, columns=areas, dtype=int)

    for i, geom_i in enumerate(gdf['geometry']):
        for j, geom_j in enumerate(gdf['geometry']):
            if i != j and geom_i.touches(geom_j):
                matrix.iloc[i, j] = 1

    return matrix

def create_NYC_closeness_adjacency_matrix(gdf, beta, e):
    areas = gdf['LocationID'].tolist()
    matrix = pd.DataFrame(0, index=areas, columns=areas, dtype=float)

    for i, geom_i in enumerate(gdf['geometry']):
        for j, geom_j in enumerate(gdf['geometry']):
            if i != j:
                distance = calculate_distance(geom_i, geom_j)

                value = np.exp(-beta * (distance ** 2))
                matrix.iloc[i, j] = value if value >= e else 0

    return matrix
# 创建连接图邻接矩阵的函数
def create_NYC_road_adjacency_matrix(gdf):
    areas = gdf['LocationID'].tolist()
    matrix = pd.DataFrame(0, index=areas, columns=areas, dtype=int)

    for i, geom_i in enumerate(gdf['geometry']):
        for j, geom_j in enumerate(gdf['geometry']):
            if i != j and geom_i.touches(geom_j):
                matrix.iloc[i, j] = 1

    return matrix
import numpy as np

# ... 你之前的代码 ...

# 定义 beta 和 epsilon 的值
beta = 1e-10 # 根据你的数据集来定义
e = 0.1 # 根据你的数据集来定义

# 计算并保存 Chicago 的邻接矩阵
Chicago_road_adjacency_matrix = create_Chicago_road_adjacency_matrix(Chicago_gdf_projected)
np.save('../data/Chicago/Chicago_road.npy', Chicago_road_adjacency_matrix)

Chicago_closeness_adjacency_matrix = create_Chicago_closeness_adjacency_matrix(Chicago_gdf_projected, beta, e)
np.save('../data/Chicago/Chicago_closeness.npy', Chicago_closeness_adjacency_matrix)

# 计算并保存 NYC 的邻接矩阵
NYC_road_adjacency_matrix = create_NYC_road_adjacency_matrix(NYC_gdf_projected)
np.save('../data/NYC/NYC_road.npy', NYC_road_adjacency_matrix)

NYC_closeness_adjacency_matrix = create_NYC_closeness_adjacency_matrix(NYC_gdf_projected, beta, e)
np.save('../data/NYC/NYC_closeness.npy', NYC_closeness_adjacency_matrix)

# 输出邻接矩阵的文件名
print("Saved files: Chicago_road.npy, Chicago_closeness.npy, NYC_road.npy, NYC_closeness.npy")