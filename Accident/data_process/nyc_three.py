import pandas as pd
import numpy as np
from shapely.geometry import Point
from shapely.wkt import loads


# 读取数据集
accident_data = pd.read_csv('../data/NYC/2016_accident/Motor_Vehicle_Collisions_-_Crashes_20240325.csv')
# 用0填充NaN值
accident_data['LOCATION'] = accident_data['LOCATION'].fillna(0)

# 删除包含NaN的行
accident_data = accident_data.dropna(subset=['LOCATION'])
# 假设嵌入维度为10
embedding_dim = 10

# 为每个小时生成嵌入向量
def generate_time_embedding(hour, embedding_dim):
    pe = np.zeros(embedding_dim)
    for pos in range(embedding_dim):
        for k in range(2 * (pos // 2) + 1):
            pe[pos] += np.power(10, -(2 * pos // 2) / embedding_dim) * np.cos(
                (k + 0.5 * hour) * np.power(2 * np.pi, pos / embedding_dim))
            if pos % 2 == 1:
                pe[pos] += np.power(10, -(2 * pos // 2) / embedding_dim) * np.sin(
                    (k + 0.5 * hour) * np.power(2 * np.pi, pos / embedding_dim))
    return pe

accident_data['CRASH_DATE_TIME'] = pd.to_datetime(accident_data['CRASH DATE'] + ' ' + accident_data['CRASH TIME'], format='%m/%d/%Y %H:%M', errors='coerce')# 获取事故时间，并转换为小时
hours = accident_data['CRASH_DATE_TIME'].dt.hour

# 创建时间嵌入矩阵
E_time = np.array([generate_time_embedding(hour, embedding_dim) for hour in hours.unique()])

# 保存时间嵌入矩阵为npy文件
np.save('../data/NYC/NYC_time.npy', E_time)

# 定义 day_name_to_index 函数
def day_name_to_index(day_name):
    week_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    return week_days.index(day_name) if pd.notnull(day_name) else None

# 定义 generate_weekday_embedding 函数
def generate_weekday_embedding(weekday_index, embedding_dim):
    pe = np.zeros(embedding_dim)
    for pos in range(embedding_dim):
        for k in range(2 * (pos // 2) + 1):
            pe[pos] += np.power(10, -(2 * pos // 2) / embedding_dim) * np.cos(
                (k + 0.5 * weekday_index) * np.power(2 * np.pi, pos / embedding_dim))
            if pos % 2 == 1:
                pe[pos] += np.power(10, -(2 * pos // 2) / embedding_dim) * np.sin(
                    (k + 0.5 * weekday_index) * np.power(2 * np.pi, pos / embedding_dim))
    return pe

# 过滤掉 'CRASH_TIME' 列中的 NaN 值
accident_data_filtered = accident_data.dropna(subset=['CRASH_DATE_TIME'])

# 获取事故星期的数值索引
days_index = accident_data_filtered['CRASH_DATE_TIME'].dt.day_name().apply(day_name_to_index)

# 创建星期嵌入矩阵，只包含非 NaN 的星期索引
E_weekday = np.array([generate_weekday_embedding(day, embedding_dim) for day in days_index if pd.notnull(day)])

# 保存星期嵌入矩阵为npy文件
np.save('NYC_weekday.npy', E_weekday)

# 读取区域数据并将 'the_geom' 列的WKT格式转换为几何对象
region_data = pd.read_csv('../data/NYC/2016_traffic/f.csv')
region_data['geometry'] = region_data['the_geom'].apply(lambda wkt: loads(wkt))

# 将事故数据中的经纬度转换为几何对象
accident_data['geometry'] = accident_data.apply(lambda row: Point(row['LONGITUDE'], row['LATITUDE']), axis=1)

# 定义 match_location_to_region 函数
def match_location_to_region(point, regions):
    for _, region in regions.iterrows():
        if point.within(region['geometry']):
            return region['LocationID']
    return None

# 为每个事故点匹配区域
location_ids = accident_data['geometry'].apply(lambda point: match_location_to_region(point, region_data))
accident_data['LocationID'] = location_ids

# 定义生成位置嵌入向量的函数
def generate_location_embedding(location_id, embedding_dim):
    pe = np.zeros(embedding_dim)
    for pos in range(embedding_dim):
        pe[pos] = (location_id + 1) * (pos + 1) / embedding_dim
    return pe

# 对每个事故的 LocationID 生成嵌入向量
location_embeddings = []

# 为每个 location_id 生成嵌入向量，并添加到列表中
for location_id in accident_data['LocationID']:
    embedding = generate_location_embedding(location_id, embedding_dim)
    location_embeddings.append(embedding)

# 将列表中的嵌入向量转换为 NumPy 数组
E_location = np.stack(location_embeddings)

# 保存地点嵌入矩阵为npy文件
np.save('NYC_location.npy', E_location)

