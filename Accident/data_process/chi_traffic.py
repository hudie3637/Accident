import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime

file_path = '../data/Chicago/2016_traffic/Taxi_Trips.csv'

# 读取数据集
data = pd.read_csv(file_path)

# 将时间戳转换为datetime类型
data['Trip Start Timestamp'] = pd.to_datetime(data['Trip Start Timestamp'], format='%m/%d/%Y %I:%M:%S %p')
data['Trip End Timestamp'] = pd.to_datetime(data['Trip End Timestamp'], format='%m/%d/%Y %I:%M:%S %p')

# 计算行程持续时间（以秒为单位）
data['trip_duration'] = (data['Trip End Timestamp'] - data['Trip Start Timestamp']).dt.total_seconds()
data = data[data['trip_duration'] > 0]  # 移除行程持续时间为0的记录

# 计算平均每小时的行程数
data['trip_hour'] = data['Trip Start Timestamp'].dt.hour

# 根据行程距离和持续时间计算平均速度，移除行程持续时间为0的记录
data['average_speed'] = data['Trip Miles'] / (data['trip_duration'] / 3600)  # 将秒转换为小时
# print(data['average_speed'])
# 选择特征列
features = ['Trip Miles', 'Pickup Community Area', 'Dropoff Community Area', 'trip_hour', 'average_speed']

# 标准化特征
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[features])

# 保存嵌入向量为npy文件
np.save('../data/Chicago/Chicago_traffic.npy', scaled_features)