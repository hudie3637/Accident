import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# 假设您的数据集存储在CSV文件中
file_path = '../data/NYC/2016_traffic/yellow_tripdata_2016_combined.csv'

# 读取数据集
data = pd.read_csv(file_path)

# 转换datetime列为datetime类型
data['tpep_pickup_datetime'] = pd.to_datetime(data['tpep_pickup_datetime'])
data['tpep_dropoff_datetime'] = pd.to_datetime(data['tpep_dropoff_datetime'])

# 计算行程持续时间（以秒为单位）
data['trip_duration'] = (data['tpep_dropoff_datetime'] - data['tpep_pickup_datetime']).dt.total_seconds()
data = data[data['trip_duration'] != 0]
# 计算平均每小时的行程数
data['trip_hour'] = data['tpep_pickup_datetime'].dt.hour

# 根据行程距离和持续时间计算平均速度
data['average_speed'] = data['trip_distance'] / (data['trip_duration'] / 3600)

# 选择特征列
features = ['trip_distance', 'PULocationID', 'DOLocationID', 'trip_hour', 'average_speed']

# 标准化特征
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[features])

# 保存嵌入向量为npy文件
np.save('../data/NYC/NYC_traffic.npy', scaled_features)