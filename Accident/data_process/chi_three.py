import pandas as pd
import numpy as np

# 读取数据集
accident_data = pd.read_csv('../data/Chicago/2016_accident/Traffic_Crashes.csv')


# 假设嵌入维度为10
embedding_dim = 10

# 为每个小时生成嵌入向量
def generate_time_embedding(hour, embedding_dim):
    if hour == 24:  # 如果小时为24，将其视为0点
        hour = 0
    pe = np.zeros(embedding_dim)
    for pos in range(embedding_dim):
        for k in range(2 * (pos // 2) + 1):
            pe[pos] += np.power(10, -(2 * pos // 2) / embedding_dim) * np.cos(
                (k + 0.5 * hour) * np.power(2 * np.pi, pos / embedding_dim))
            if pos % 2 == 1:
                pe[pos] += np.power(10, -(2 * pos // 2) / embedding_dim) * np.sin(
                    (k + 0.5 * hour) * np.power(2 * np.pi, pos / embedding_dim))
    return pe

# 确保CRASH_DATE列是字符串格式
accident_data['CRASH_DATE'] = accident_data['CRASH_DATE'].astype(str)

# 转换CRASH_DATE列为datetime类型，过滤掉无法解析的日期时间
accident_data['CRASH_TIME'] = pd.to_datetime(accident_data['CRASH_DATE'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

# 删除包含NaT值的行
accident_data = accident_data[~accident_data['CRASH_TIME'].isna()]

# 提取小时
hours = accident_data['CRASH_TIME'].dt.hour

# 创建时间嵌入矩阵，跳过24小时的情况（如果需要）
E_time = np.array([generate_time_embedding(hour, embedding_dim) for hour in hours if hour != 24])

# 保存时间嵌入矩阵为npy文件
np.save('Chicago_time.npy', E_time)

# 打印结果
print(E_time)
# # 定义 day_name_to_index 函数
# def day_name_to_index(day_name):
#     week_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
#     return week_days.index(day_name)  # 直接返回星期几的数值索引
#
# # 定义生成星期嵌入的函数
# def generate_weekday_embedding(weekday_index, embedding_dim):
#     pe = np.zeros(embedding_dim)
#     for pos in range(embedding_dim):
#         for k in range(2 * (pos // 2) + 1):
#             pe[pos] += np.power(10, -(2 * pos // 2) / embedding_dim) * np.cos(
#                 (k + 0.5 * weekday_index) * np.power(2 * np.pi, pos / embedding_dim))
#             if pos % 2 == 1:
#                 pe[pos] += np.power(10, -(2 * pos // 2) / embedding_dim) * np.sin(
#                     (k + 0.5 * weekday_index) * np.power(2 * np.pi, pos / embedding_dim))
#     return pe
# accident_data['CRASH_DATE'] = pd.to_datetime(accident_data['CRASH_DATE'], format='%m/%d/%Y %I:%M:%S %p')
# # 过滤掉 'CRASH_TIME' 列中的 NaN 值
# accident_data_filtered = accident_data.dropna(subset=['CRASH_DATE'])
#
#
# # 获取事故星期的数值索引
# days_index = accident_data_filtered['CRASH_DATE'].dt.day_name().apply(day_name_to_index)
# # 创建星期嵌入矩阵，只包含非 NaN 的星期索引
# E_weekday = np.array([generate_weekday_embedding(day, embedding_dim) for day in days_index])
# # 保存星期嵌入矩阵为npy文件
# np.save('Chicago_weekday.npy', E_weekday)
#
# # 读取区域数据并将 'the_geom' 列的WKT格式转换为几何对象
# region_data = pd.read_csv('../data/Chicago/raw/f.csv')
# region_data['geometry'] = region_data['the_geom'].apply(lambda wkt: loads(wkt))
#
# # 将事故数据中的经纬度转换为几何对象
# accident_data['geometry'] = accident_data.apply(lambda row: Point(row['LONGITUDE'], row['LATITUDE']), axis=1)
#
# # 定义 match_location_to_region 函数
# def match_location_to_region(point, regions):
#     for _, region in regions.iterrows():
#         if point.within(region['geometry']):
#             return region['AREA_NUMBE']
#     return None
#
# # 为每个事故点匹配区域
# location_ids = accident_data['geometry'].apply(lambda point: match_location_to_region(point, region_data))
# accident_data['AREA_NUMBE'] = location_ids
#
# # 定义生成位置嵌入向量的函数
# def generate_location_embedding(location_id, embedding_dim):
#     pe = np.zeros(embedding_dim)
#     for pos in range(embedding_dim):
#         pe[pos] = (location_id + 1) * (pos + 1) / embedding_dim
#     return pe
#
# embeddings_dict = {}
# for location_id in accident_data['AREA_NUMBE'].unique():
#     if pd.notnull(location_id):
#         embeddings_dict[location_id] = generate_location_embedding(location_id, embedding_dim)
#
# # 应用已有的嵌入向量
# accident_data['location_embedding'] = accident_data['AREA_NUMBE'].apply(lambda x: embeddings_dict.get(x, np.zeros(embedding_dim)))
#
# # 将生成的嵌入向量转换为适合保存的格式
# E_location = np.stack(accident_data['location_embedding'].values)
#
# # 保存地点嵌入矩阵为npy文件
# np.save('Chicago_location.npy', E_location)