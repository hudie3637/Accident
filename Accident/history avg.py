import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# 加载数据
df = pd.read_csv("data/Chicago/2016_traffic/Taxi_Trips.csv")

# 确保时间戳列是 datetime 类型
df['Trip Start Timestamp'] = pd.to_datetime(df['Trip Start Timestamp'], errors='coerce')
df['Trip End Timestamp'] = pd.to_datetime(df['Trip End Timestamp'], errors='coerce')
df['Trip Start Date'] = df['Trip Start Timestamp'].dt.date
df['Trip Start Hour'] = df['Trip Start Timestamp'].dt.hour

# 计算速度，确保 'Trip Miles' 和 'Trip Seconds' 是数值类型
df['Trip Miles'] = pd.to_numeric(df['Trip Miles'], errors='coerce')
df['Trip Seconds'] = pd.to_numeric(df['Trip Seconds'], errors='coerce')
df['Speed'] = df['Trip Miles']*1000 / df['Trip Seconds']

# 聚合数据，计算每小时的平均速度
hourly_speeds = df.groupby(['Trip Start Date', 'Trip Start Hour'])['Speed'].mean().reset_index()
hourly_speeds.dropna(inplace=True)

# 确保 'Speed' 是数值类型
hourly_speeds['Speed'] = pd.to_numeric(hourly_speeds['Speed'], errors='coerce')

# 构建二维数组，其中包含 'Speed'
df_array = hourly_speeds[['Speed']].values

# 准备数据
window_size = 24
history_length = window_size  # 使用的历史数据长度

# 创建日期列以便于可视化
hourly_speeds['date'] = pd.to_datetime(hourly_speeds['Trip Start Date'].astype(str) + ' ' + hourly_speeds['Trip Start Hour'].astype(str) + ':00:00')
dates = hourly_speeds['date'].values
print(hourly_speeds)
# 分割数据为训练集和测试集
train_data = df_array[:-window_size].flatten()
test_data = df_array[-window_size:].flatten()

# 定义历史平均值基线模型
class HistoryAverageBaseline:
    def __init__(self, window_size=24):
        self.window_size = window_size
        self.history = []

    def fit(self, data):
        # 保存历史数据
        self.history = data[-self.window_size:]

    def predict(self, n_steps):
        # 预测未来 n_steps 的值
        if len(self.history) == 0:
            raise ValueError("Model has not been fitted with data.")
        history_mean = np.mean(self.history)
        predictions = [history_mean] * n_steps
        return np.array(predictions)

# 初始化并训练基线模型
baseline_model = HistoryAverageBaseline(window_size=history_length)
baseline_model.fit(train_data)

# 进行预测
n_steps = len(test_data)
predictions = baseline_model.predict(n_steps=n_steps)

# 计算评估指标
mse = mean_squared_error(test_data, predictions)
mae = mean_absolute_error(test_data, predictions)

print(f"History Average Baseline MSE: {mse}")
print(f"History Average Baseline MAE: {mae}")

# 可视化结果
plt.figure(figsize=(12, 6))
plt.plot(dates[-window_size:], test_data, label='True Values')
plt.plot(dates[-window_size:], predictions, label='Predictions', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Speed')
plt.title('History Average Baseline Predictions')
plt.legend()
plt.show()
