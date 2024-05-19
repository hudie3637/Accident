import numpy as np
import torch
from sklearn.model_selection import train_test_split

def load_data(data_path):
    try:
        # 加载数据
        X_BERT = np.load(f'{data_path}/Chicago_BERT_features.npy')
        X_weekday = np.load(f'{data_path}/Chicago_weekday.npy')
        X_time = np.load(f'{data_path}/Chicago_time.npy')
        X_location = np.load(f'{data_path}/Chicago_location.npy')
        traffic_data = np.load(f'{data_path}/Chicago_traffic_weather_sequence.npy')
        y = np.load(f'{data_path}/Chicago_severity_labels.npy') - 1
        # 检查交通数据中是否包含 NaN
        if np.isnan(traffic_data).any():
            print("NaN found in traffic_data")
            # 处理 NaN，例如使用列的平均值填充
            traffic_data = np.nan_to_num(traffic_data, nan=traffic_data.mean())

        # 将交通数据转换为正确的形状
        X_traffic = traffic_data.reshape(-1, 1, traffic_data.shape[1])

        X_traffic = X_traffic.squeeze(1)

        # 将数据转换为 PyTorch 张量
        X_BERT_tensor = torch.tensor(X_BERT, dtype=torch.float32)
        X_weekday_tensor = torch.tensor(X_weekday, dtype=torch.float32)
        X_time_tensor = torch.tensor(X_time, dtype=torch.float32)
        X_location_tensor = torch.tensor(X_location, dtype=torch.float32)
        X_traffic_tensor = torch.tensor(X_traffic, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        # 检查张量中的 NaN
        if torch.isnan(X_traffic_tensor).any():
            print("NaN found in X_traffic_tensor")

        return X_BERT_tensor, X_weekday_tensor, X_time_tensor, X_location_tensor, X_traffic_tensor, y_tensor
    except IOError as e:
        print(f'Error loading data: {e}')
        return None

# 定义你的数据路径
data_path = '../data/Chicago'

# 调用函数
results = load_data(data_path)