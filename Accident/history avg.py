import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("data/Chicago/2016_traffic/Taxi_Trips.csv")

# 将时间戳列转换为 datetime 类型
df['Trip Start Timestamp'] = pd.to_datetime(df['Trip Start Timestamp'], errors='coerce')
df['Trip End Timestamp'] = pd.to_datetime(df['Trip End Timestamp'], errors='coerce')

# 计算行程开始日期和小时
df['Trip Start Date'] = df['Trip Start Timestamp'].dt.date
df['Trip Start Hour'] = df['Trip Start Timestamp'].dt.hour

# 确保 'Trip Miles' 和 'Trip Seconds' 为数值类型，并计算速度
df['Trip Miles'] = pd.to_numeric(df['Trip Miles'], errors='coerce')
df['Trip Seconds'] = pd.to_numeric(df['Trip Seconds'], errors='coerce')

# 避免除以零的错误
df['Speed'] = 0
df.loc[df['Trip Seconds'] > 0, 'Speed'] = df['Trip Miles'] * 1000 / df['Trip Seconds']

# 聚合数据，计算每个社区区域在每个时间点的平均速度
hourly_speeds = df.groupby(['Pickup Community Area', 'Trip Start Date', 'Trip Start Hour'])['Speed'].mean().reset_index()
# 确保hourly_speeds按时间排序
hourly_speeds = hourly_speeds.sort_values(['Trip Start Date', 'Trip Start Hour'])
# 创建一个唯一的MultiIndex
index = pd.MultiIndex.from_frame(hourly_speeds[['Trip Start Date', 'Trip Start Hour']].drop_duplicates())

# 创建一个新的DataFrame，包含77列，每列对应一个社区区域
speed_df = pd.DataFrame(0.0, index=index, columns=hourly_speeds['Pickup Community Area'].unique())

# 填充DataFrame的值
for _, row in hourly_speeds.iterrows():
    speed_df.loc[(row['Trip Start Date'], row['Trip Start Hour']), row['Pickup Community Area']] = row['Speed']

# 填充缺失值
speed_df = speed_df.fillna(0).astype(float)
print(f'speed_df: {speed_df.shape}')
df_array = speed_df.values
print(f'df_array: {df_array.shape}')
# 设置窗口大小
window_size = 24  # 每个时间点考虑的前24小时数据

# 初始化用于存储预测值的数组
predictions = np.zeros((df_array.shape[0] - window_size, df_array.shape[1]))
# 计算每个时间点前24小时的平均速度
for i in range(window_size, df_array.shape[0]):
    # 选择当前时间点前24小时的数据
    window_data = df_array[i - window_size:i]
    # 计算平均值，注意axis=1是列方向，因为我们是按列存储每个时间点的数据
    predictions[i - window_size] = np.nanmean(window_data, axis=0)

# 真实值
actuals = df_array[window_size:]

# 计算MSE和MAE
mse = mean_squared_error(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)

print(f"Historical Average Model Performance:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")

# 绘制第一个社区区域的预测速度与实际速度对比图
plt.figure(figsize=(12, 6))
plt.plot(range(len(actuals[:, 0])), actuals[:, 0], label='Actual Average Speeds')
plt.plot(range(len(predictions[:, 0])), predictions[:, 0], label='Predicted Speeds')
plt.axhline(y=np.mean(actuals[:, 0]), color='r', linestyle='--', label='Overall Average Speed')
plt.legend()
plt.title('Historical Average Model Predictions vs Actual Speeds for Community Area 1')
plt.xlabel('Time Window')
plt.ylabel('Average Speed (m/s)')
plt.show()
