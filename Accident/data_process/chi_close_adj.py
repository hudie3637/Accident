import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely import LineString
from shapely.ops import nearest_points
from shapely.wkt import loads
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
pd.set_option('future.no_silent_downcasting', True)
# 读取CSV文件
data = pd.read_csv('../data/Chicago/raw/CommAreas.csv')
# 删除包含缺失值的行
data = data.dropna()
# 删除重复的行
data = data.drop_duplicates()
# 解析几何信息，创建GeoDataFrame
data['geometry'] = data['the_geom'].apply(lambda wkt_str: loads(wkt_str))
gdf = gpd.GeoDataFrame(data, geometry='geometry', crs='EPSG:4326')
# 将几何数据从地理CRS转换到投影CRS
gdf_projected = gdf.to_crs(crs='EPSG:3857')
# 计算区域之间的距离
# 创建一个空的DataFrame作为距离矩阵
distance_matrix = pd.DataFrame(index=gdf['AREA_NUMBE'], columns=gdf['AREA_NUMBE'])
# 计算每对区域之间的距离并填充距离矩阵
for i in range(len(gdf)):
    for j in range(i+1, len(gdf)):
        distance_sum = 0
        for poly_i in gdf.loc[i, 'geometry'].geoms:
            for poly_j in gdf.loc[j, 'geometry'].geoms:
                try:
                    distance_sum += poly_i.distance(poly_j)
                except Exception as e:
                    pass
        distance_matrix.at[gdf['AREA_NUMBE'][i], gdf['AREA_NUMBE'][j]] = distance_sum
        distance_matrix.at[gdf['AREA_NUMBE'][j], gdf['AREA_NUMBE'][i]] = distance_sum
# 移除 NaN 值和无穷大值，并保留对象类型
distance_matrix = distance_matrix.replace([np.inf, -np.inf], np.nan).fillna(0).infer_objects(copy=False)
# 计算最小值和最大值，确保它们是有效的数值
min_dist = distance_matrix.min().min()
max_dist = distance_matrix.max().max()

# 应用颜色映射
color_map = plt.get_cmap('viridis')
norm = plt.Normalize(vmin=min_dist, vmax=max_dist)
mapper = plt.cm.ScalarMappable(norm=norm, cmap=color_map)
mapper.set_array([])  # 初始化 ScalarMappable 对象
# 创建一个新的图形窗口，并设置大小
fig, ax = plt.subplots(figsize=(15, 15))
# 为了避免 FutureWarning，我们可以设置 Pandas 的选项
pd.set_option('future.no_silent_downcasting', True)
# 解析几何信息，创建GeoDataFrame
data['geometry'] = data['the_geom'].apply(lambda wkt_str: loads(wkt_str))

gdf.plot(ax=ax, edgecolor='black', linewidth=0.5, legend=True, cmap='viridis', column='AREA_NUMBE')
# 绘制社区区域的多边形
#gdf.plot(ax=ax, edgecolor='black', linewidth=0.5,  legend=True, cmap='viridis', column='AREA_NUMBE')
# 确定相邻区域并创建一个邻接列表
adjacent_areas = []
for i in range(len(gdf)):
    for j in range(i + 1, len(gdf)):
        if gdf['geometry'].iloc[i].touches(gdf['geometry'].iloc[j]):
            adjacent_areas.append((gdf['AREA_NUMBE'].iloc[i], gdf['AREA_NUMBE'].iloc[j]))
# 在每个多边形中心添加文本标签
for area_numbe, row in gdf.iterrows():
    center = row['geometry'].centroid
    ax.text(center.x, center.y, str(row['AREA_NUMBE']), fontsize=7, ha='center', va='center', color='black')

# 绘制相邻区域之间的连线
for start_area, end_area in adjacent_areas:
    # 获取起始和结束区域的中心点
    start_point = gdf[gdf['AREA_NUMBE'] == start_area]['geometry'].centroid.iloc[0]
    end_point = gdf[gdf['AREA_NUMBE'] == end_area]['geometry'].centroid.iloc[0]
    # 创建LineString对象表示连接线
    line = LineString([start_point, end_point])
    # 绘制这条连接线
    ax.plot(line.coords.xy[0], line.coords.xy[1], color='gray', linewidth=1, linestyle='-')

# 设置图表标题和坐标轴标签
ax.set_title('Closeness Graph of Chicago Communities with Road Network')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# 显示图表
plt.show()

