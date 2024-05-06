import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely import LineString, Polygon
from shapely.ops import nearest_points
from shapely.wkt import loads
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point

# 设置Pandas的选项
pd.set_option('future.no_silent_downcasting', True)

# 读取CSV文件
data = pd.read_csv('../data/NYC/taxi_zones.csv')

# 删除包含缺失值的行
data.dropna(subset=['the_geom', 'LocationID'], inplace=True)

# 删除重复的行
data.drop_duplicates(inplace=True)

# 解析几何信息，创建GeoDataFrame
data['geometry'] = data['the_geom'].apply(lambda wkt_str: loads(wkt_str))
gdf = gpd.GeoDataFrame(data, geometry='geometry', crs='EPSG:4326')
if 'LocationID' not in gdf.columns:
    raise ValueError("The 'LocationID' column does not exist in the GeoDataFrame. Please check the column names.")
# 计算区域之间的邻接关系
adjacent_areas = []
for i in range(len(gdf)):
    for j in range(i + 1, len(gdf)):
        if gdf.iloc[i].geometry.intersects(gdf.iloc[j].geometry):
            adjacent_areas.append((gdf['LocationID'].iloc[i], gdf['LocationID'].iloc[j]))

# 创建一个NetworkX图
G = nx.Graph()

# 添加邻接关系
for start_area, end_area in adjacent_areas:
    G.add_edge(start_area, end_area)

# 计算区域之间的距离矩阵
distance_matrix = pd.DataFrame(index=gdf['LocationID'], columns=gdf['LocationID'])
for i in range(len(gdf)):
    for j in range(i+1, len(gdf)):
        if gdf.iloc[i].geometry.intersects(gdf.iloc[j].geometry):
            distance_matrix.at[gdf['LocationID'].iloc[i], gdf['LocationID'].iloc[j]] = gdf.iloc[i].geometry.distance(gdf.iloc[j].geometry)

# 移除NaN值和无穷大值，并保留对象类型
distance_matrix = distance_matrix.replace([np.inf, -np.inf], np.nan).fillna(0).infer_objects(copy=False)

# 计算最小值和最大值，确保它们是有效的数值
min_dist = distance_matrix.min().min()
max_dist = distance_matrix.max().max()

# 应用颜色映射
color_map = plt.get_cmap('viridis')
norm = plt.Normalize(vmin=min_dist, vmax=max_dist)
mapper = plt.cm.ScalarMappable(norm=norm, cmap=color_map)
mapper.set_array([])  # 初始化 ScalarMappable 对象

# 绘制社区区域的多边形，并应用颜色映射
fig, ax = plt.subplots(figsize=(15, 15))
# gdf['color'] = mapper.to_rgba(distance_matrix[gdf['LocationID'], gdf['LocationID']])  # 应用颜色到每个区域
gdf.plot(ax=ax, edgecolor='black', linewidth=0.5, legend=True, cmap='viridis', column='LocationID')

# 在每个多边形中心添加文本标签
for area_numbe, row in gdf.iterrows():
    center = row['geometry'].centroid
    ax.text(center.x, center.y, str(row['LocationID']), fontsize=5, ha='center', va='center', color='white')
# 绘制相邻区域之间的连线
for start_area, end_area in adjacent_areas:
    # 获取起始和结束区域的中心点
    start_point = gdf[gdf['LocationID'] == start_area]['geometry'].centroid.iloc[0]
    end_point = gdf[gdf['LocationID'] == end_area]['geometry'].centroid.iloc[0]
    # 创建LineString对象表示连接线
    line = LineString([start_point, end_point])
    # 绘制这条连接线
    ax.plot(line.coords.xy[0], line.coords.xy[1], color='gray', linewidth=1, linestyle='-')
# 设置图表标题和坐标轴标签
ax.set_title('Closeness Graph of New York Taxi Zones')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# 显示图表
plt.show()