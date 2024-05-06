import geopandas as gpd
import pandas as pd
from shapely.wkt import loads
# location_map = pd.read_csv('../data/NYC/2016_traffic/f.csv')
# trips_data = pd.read_csv('../data/NYC/2016_traffic/yellow_tripdata_2016-06.csv')
# # 清理并转换数据类型
# location_map.dropna(subset=['the_geom', 'LocationID'], inplace=True)
# location_map['LocationID'] = location_map['LocationID'].astype(int)
# location_map['the_geom'] = location_map['the_geom'].apply(lambda x: loads(x))
# location_gdf = gpd.GeoDataFrame(location_map, geometry='the_geom')
#
# # 转换经纬度数据为几何类型
# trips_data[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']] = trips_data[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].apply(pd.to_numeric, errors='coerce')
# pickup_points = gpd.points_from_xy(trips_data['pickup_longitude'], trips_data['pickup_latitude'])
# dropoff_points = gpd.points_from_xy(trips_data['dropoff_longitude'], trips_data['dropoff_latitude'])
#
# pickup_points_gdf = gpd.GeoDataFrame(trips_data, geometry=pickup_points)
# dropoff_points_gdf = gpd.GeoDataFrame(trips_data, geometry=dropoff_points)
# pickup_sjoin = gpd.sjoin(pickup_points_gdf, location_gdf, how='left', predicate='within')
# dropoff_sjoin = gpd.sjoin(dropoff_points_gdf, location_gdf, how='left', predicate='within')
# # 将上车地点的 LocationID 添加到原始数据集
# trips_data['PULocationID'] = pickup_sjoin['LocationID']
#
# # 将下车地点的 LocationID 添加到原始数据集
# trips_data['DOLocationID'] = dropoff_sjoin['LocationID']
# # 删除PULocationID或DOLocationID为空的行
# trips_data.dropna(subset=['PULocationID', 'DOLocationID'], inplace=True)
# trips_data.drop(columns=['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'], inplace=True)# # 确认删除后的数据集大小
# # print(trips_data.shape)
#
# # 保存更新后的数据集为CSV文件
# trips_data.to_csv('data/NYC/2016_traffic/yellow_tripdata_2016-06_with_location_cleaned.csv', index=False)
# 定义文件路径模式
import pandas as pd
import os

# 定义文件夹路径
data_folder = '../data/NYC/2016_traffic/'  # 请替换为包含数据文件的实际文件夹路径

# 获取所有CSV文件的列表
csv_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if
             f.startswith('yellow_tripdata_2016') and f.endswith('.csv')]

# 初始化一个空的DataFrame用于存储合并后的数据
combined_data = pd.DataFrame()

# 循环读取每个CSV文件并合并到combined_data DataFrame
for file in csv_files:
    # 读取当前月份的数据
    month_data = pd.read_csv(file)

    # 如果combined_data是空的DataFrame，直接赋值
    if combined_data.empty:
        combined_data = month_data
    else:
        # 合并数据，使用ignore_index=True以重置索引
        combined_data = pd.concat([combined_data, month_data], ignore_index=True)

# 检查combined_data的类型，确保它仍然是一个DataFrame对象
if not isinstance(combined_data, pd.DataFrame):
    raise ValueError("combined_data is not a DataFrame object after merging.")

# 重置索引，因为多次合并可能会导致索引问题
combined_data.reset_index(drop=True, inplace=True)
print(combined_data)
# 保存合并后的数据到新的CSV文件
output_file = os.path.join(data_folder, 'yellow_tripdata_2016_combined.csv')
combined_data.to_csv(output_file, index=False)

print("Merging process completed successfully. Combined data saved to CSV file.")