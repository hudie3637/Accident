import pandas as pd
import numpy as np

def load_and_process_data():
    # 加载交通数据
    traffic_data = pd.read_csv('../data/Chicago/2016_traffic/Taxi_Trips.csv')
    traffic_data['hour'] = pd.to_datetime(traffic_data['Trip Start Timestamp']).dt.floor('H')

    # 加载事故数据
    accident_data = pd.read_csv('../data/Chicago/2016_accident/Traffic_Crashes.csv')
    accident_data['hour'] = pd.to_datetime(accident_data['CRASH_DATE']).dt.floor('H')

    # 聚合交通数据以匹配事故数据的时间戳
    traffic_data_grouped = traffic_data.groupby('hour').mean()  # 或使用其他适当的聚合逻辑

    # 将交通数据映射到事故数据
    accident_data = accident_data.join(traffic_data_grouped, on='hour', rsuffix='_traffic')

    # 保存处理后的数据
    accident_data.to_csv('Chicago_processed_accident_data.csv', index=False)

    # 验证维度
    print(accident_data.shape)  # 应与其他特征数据的样本数匹配

if __name__ == "__main__":
    load_and_process_data()
