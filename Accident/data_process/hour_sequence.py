import pandas as pd
import numpy as np

# 加载交通流数据
traffic_data_path = '../data/Chicago/2016_accident/Chicago_processed_accident_data.csv'
traffic_df = pd.read_csv(traffic_data_path)

# 确保CRASH_DATE列是datetime类型
traffic_df['CRASH_DATE'] = pd.to_datetime(traffic_df['CRASH_DATE'])

# 设置CRASH_DATE列为索引
traffic_df.set_index('CRASH_DATE', inplace=True)

# 加载天气数据
weather_data_path = '../data/Chicago/Chicago_weather.npy'
weather_data = np.load(weather_data_path, allow_pickle=True)

# 假设weather_data中的每个元素都是一个字典
column_names = ['valid'] + ['col_{}'.format(i) for i in range(1, 90)]
weather_df = pd.DataFrame(weather_data, columns=column_names)
weather_df['valid'] = pd.to_datetime(weather_df['valid'])

# 确保traffic_df和weather_df使用统一的时间戳格式
traffic_df['pickup_hour'] = traffic_df.index.floor('H')

# 创建时间范围
start_hour = pd.to_datetime('2016-01-01 00:00')
end_hour = pd.to_datetime('2016-12-31 23:00')
all_hours = pd.date_range(start=start_hour, end=end_hour, freq='H')

# 创建空DataFrame并填充天气和交通数据
df = pd.DataFrame(index=all_hours, columns=['tmpf', 'dwpf', 'relh', 'sknt', 'mslp', 'trip_counts'])
window_size = 24

# 填充天气数据
for _, row in weather_df.iterrows():
    if row['valid'] in df.index:
        df.loc[row['valid'], ['tmpf', 'dwpf', 'relh', 'sknt', 'mslp']] = row[['col_1', 'col_2', 'col_3', 'col_4', 'col_5']]

# 填充交通流数据
traffic_counts = traffic_df.resample('H').size().reindex(all_hours, fill_value=0)
df['trip_counts'] = traffic_counts
# 数据归一化前，处理NaN值
df.fillna(0, inplace=True)  # 用0填充NaN值
# 数据归一化
df = df.astype(float)  # 确保所有列都是数值类型，以便进行归一化
df_scaled = (df - df.min()) / (df.max() - df.min()).replace({np.nan: 0, np.inf: 0})
# 为Conv1d准备数据前，再次检查NaN值
df_scaled.fillna(0, inplace=True)
# 为Conv1d准备数据
X = []
for i in range(len(df_scaled) - window_size):
    X.append(df_scaled.iloc[i:i+window_size].values)
X_reshaped = np.array(X)

# 保存数据
np.save('Chicago_traffic_weather_sequence.npy', X_reshaped)
print(X_reshaped)