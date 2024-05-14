import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("data/Chicago/2016_traffic/Taxi_Trips.csv")

# 将时间戳列转换为 datetime 类型
df['Trip Start Timestamp'] = pd.to_datetime(df['Trip Start Timestamp'], errors='coerce')
df['Trip End Timestamp'] = pd.to_datetime(df['Trip End Timestamp'], errors='coerce')

# 过滤数据
df = df.dropna(subset=['Trip Miles', 'Trip Seconds'])
df = df[(df['Trip Miles'] > 0) & (df['Trip Seconds'] > 0)]

# 计算速度
df['Trip Miles_km'] = df['Trip Miles'] * 1.60934
df['Trip Seconds_h'] = df['Trip Seconds'] / 3600
df['Speed'] = df['Trip Miles_km'] / df['Trip Seconds_h']

# 过滤异常速度值
reasonable_speed_threshold = 200
df = df[df['Speed'] <= reasonable_speed_threshold]

# 定义函数生成时间点范围
def generate_time_range(row):
    return pd.date_range(start=row['Trip Start Timestamp'],
                         end=row['Trip End Timestamp'],
                         freq='15T')  # 15分钟间隔

# 对每个行程生成时间点范围
df['Time Points'] = df.apply(generate_time_range, axis=1)

# 将DataFrame展开为长格式，每个时间点一行
expanded_df = df.explode('Time Points')

# 提取时间点的日期和小时
expanded_df['Time Point Date'] = expanded_df['Time Points'].dt.date
expanded_df['Time Point Hour'] = expanded_df['Time Points'].dt.hour
expanded_df['Time Point Minute'] = expanded_df['Time Points'].dt.minute

# 使用 'Pickup Community Area' 作为区域信息
result_df = expanded_df[
    ['Time Points', 'Time Point Date', 'Time Point Hour', 'Time Point Minute', 'Pickup Community Area', 'Speed']
]

# 聚合处理重复项，确保唯一
aggregated_df = result_df.groupby(['Time Points', 'Time Point Date', 'Time Point Hour', 'Time Point Minute', 'Pickup Community Area']).agg({'Speed': 'mean'}).reset_index()

# 将数据转换为每行一个时间点，每列一个区域的DataFrame
final_df = aggregated_df.pivot_table(index=['Time Point Date', 'Time Point Hour', 'Time Point Minute'],
                                     columns='Pickup Community Area',
                                     values='Speed',
                                     fill_value=0)

# 转换为 numpy 数组
df_array = final_df.values
df_array = np.nan_to_num(df_array)  # 确保没有 NaN 值

# 设置窗口大小
window_size = 24 * 4  # 每15分钟间隔的24小时

# 初始化用于存储预测值的数组
predictions = np.zeros((df_array.shape[0] - window_size, df_array.shape[1]))

# 计算每个时间点前24小时的平均速度
for i in range(window_size, df_array.shape[0]):
    window_data = df_array[i - window_size:i, :]
    predictions[i - window_size] = np.nanmean(window_data, axis=0)

# 真实值
actuals = df_array[window_size:]
# 绘制第一个社区区域的速度直方图
plt.figure(figsize=(10, 6))
plt.hist(actuals[:, 0], bins=100, color='blue', edgecolor='black')
plt.title('Speed Distribution for Community Area 1')  # Adjust the title to match the area index
plt.xlabel('Speed (km/h)')
plt.ylabel('Frequency')
plt.show()
# 计算MSE和MAE
mse = mean_squared_error(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)

print(f"Historical Average Model Performance:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")

# 绘制预测速度与实际速度对比图
plt.figure(figsize=(12, 6))
plt.plot(predictions[:, 0], label='Predicted Speeds for Community Area 1', color='orange')
plt.plot(actuals[:, 0], label='Actual Speeds for Community Area 1', color='blue')
plt.axhline(y=np.mean(actuals[:, 0]), color='r', linestyle='--', label=f'Overall Average Speed of Community Area 1')
plt.legend()
plt.title('Historical Average Model Predictions vs Actual Speeds for Community Area 1')
plt.xlabel('Time Window')
plt.ylabel('Average Speed (km/h)')
plt.show()
