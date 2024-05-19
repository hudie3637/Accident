import numpy as np
import torch
from sklearn.model_selection import train_test_split

def load_data(data_path):
    try:
        # 加载数据
        # X_BERT = np.load(f'{data_path}/Chicago_BERT_features.npy')
        # X_weekday = np.load(f'{data_path}/Chicago_weekday.npy')
        # X_time = np.load(f'{data_path}/Chicago_time.npy')
        # X_location = np.load(f'{data_path}/Chicago_location.npy')
        # traffic_data = np.load(f'{data_path}/Chicago_traffic_weather_sequence.npy')
        # y = np.load(f'{data_path}/Chicago_severity_labels.npy') - 1
        X_BERT = np.load(f'{data_path}/NYC_BERT_features.npy')
        X_weekday = np.load(f'{data_path}/NYC_weekday.npy')
        X_time = np.load(f'{data_path}/NYC_time.npy')
        X_location = np.load(f'{data_path}/NYC_location.npy')
        traffic_data = np.load(f'{data_path}/NYC_traffic_weather_sequence.npy')
        y = np.load(f'{data_path}/NYC_severity_labels.npy') - 1
        # 将交通数据转换为正确的形状
        X_traffic = traffic_data.reshape(-1, 1, traffic_data.shape[1])

        X_traffic = X_traffic.squeeze(1)
        X_BERT = X_BERT[:50430]
        X_weekday = X_weekday[:50430]
        X_time = X_time[:50430]
        X_traffic=X_traffic[:50430]
        # 将数据转换为PyTorch张量
        X_BERT_tensor = torch.tensor(X_BERT, dtype=torch.float32)
        X_weekday_tensor = torch.tensor(X_weekday, dtype=torch.float32)
        X_time_tensor = torch.tensor(X_time, dtype=torch.float32)
        X_location_tensor = torch.tensor(X_location, dtype=torch.float32)
        X_traffic_tensor = torch.tensor(X_traffic, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        return X_BERT_tensor, X_weekday_tensor, X_time_tensor, X_location_tensor, X_traffic_tensor, y_tensor
    except IOError as e:
        print(f'Error loading data: {e}')
        return None