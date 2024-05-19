import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Reading data
df = pd.read_csv("data/Chicago/2016_traffic/Taxi_Trips.csv")

# Converting timestamp columns to datetime
df['Trip Start Timestamp'] = pd.to_datetime(df['Trip Start Timestamp'], errors='coerce')
df['Trip End Timestamp'] = pd.to_datetime(df['Trip End Timestamp'], errors='coerce')

# Filtering data for valid entries
df = df.dropna(subset=['Trip Miles', 'Trip Seconds'])
df = df[(df['Trip Miles'] > 0) & (df['Trip Seconds'] > 0)]

# Calculating speed
df['Trip Miles_km'] = df['Trip Miles'] * 1.60934  # converting miles to km
df['Trip Seconds_h'] = df['Trip Seconds'] / 3600  # converting seconds to hours
df['Speed'] = df['Trip Miles_km'] / df['Trip Seconds_h']

# Filtering out unreasonable speed values
reasonable_speed_threshold = 200
df = df[df['Speed'] <= reasonable_speed_threshold]

# Creating time ranges for each trip
df['Time Points'] = df.apply(lambda row: pd.date_range(start=row['Trip Start Timestamp'],
                                                      end=row['Trip End Timestamp'],
                                                      freq='15T'), axis=1)

# Expanding DataFrame to have one row per time point
expanded_df = df.explode('Time Points')
expanded_df['Time Point Date'] = expanded_df['Time Points'].dt.date
expanded_df['Time Point Hour'] = expanded_df['Time Points'].dt.hour
expanded_df['Time Point Minute'] = expanded_df['Time Points'].dt.minute

# Aggregating by average speed per community area and time point
result_df = expanded_df.groupby(['Time Point Date', 'Time Point Hour', 'Time Point Minute', 'Pickup Community Area']).agg({'Speed': 'mean'}).reset_index()

# Pivoting the DataFrame
final_df = result_df.pivot_table(index=['Time Point Date', 'Time Point Hour', 'Time Point Minute'],
                                 columns='Pickup Community Area', values='Speed', fill_value=0)

# Function to fill zeros with the mean of the previous day
def fill_with_previous_day_mean(df):
    for column in df.columns:
        # Calculating the mean speed of the previous day for each hour and minute
        df[column] = df[column].replace(0, pd.NA).fillna(method='ffill').fillna(method='bfill')
    return df

final_df = fill_with_previous_day_mean(final_df)

# Function to fill remaining zeros using a rolling mean
def fill_with_rolling_mean(df, window_size=7):
    for column in df.columns:
        df[column] = df[column].rolling(window=window_size, min_periods=1).mean().fillna(method='bfill')
    return df

final_df = fill_with_rolling_mean(final_df)

# Calculate global mean and fill any remaining missing values
global_mean = final_df.mean().mean()  # average speed across all times and areas
final_df = final_df.fillna(global_mean)

# Preparing the data array for predictions
df_array = final_df.values
# Set the window size for historical averages
window_size = 24
# Initialize an array for predictions
predictions = np.zeros((df_array.shape[0] - window_size, df_array.shape[1]))

# Calculate the average speed for the previous 24 time points
for i in range(window_size, df_array.shape[0]):
    window_data = df_array[i - window_size:i, :]
    predictions[i - window_size] = np.nanmean(window_data, axis=0)

# Actual speeds after the window
actuals = df_array[window_size:]

# Compute MAE, MAPE, and RMSE
mae = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mean_squared_error(actuals, predictions))
mape = np.mean(np.abs((actuals - predictions) / actuals)[actuals > 0])
# 计算损失
loss = mean_squared_error(actuals, predictions)


# Output results
print(f"Historical Average Model Performance:")
print(f"MAE: {mae}")
print(f"MAPE: {mape}")
print(f"RMSE: {rmse}")
print(f"Loss: {loss}")
# You can add plotting code here if needed

# 绘制第一个社区区域的速度直方图
plt.figure(figsize=(10, 6))
plt.hist(actuals[:, 0], bins=100, color='blue', edgecolor='black')
plt.title('Speed Distribution for Community Area 1')  # Adjust the title to match the area index
plt.xlabel('Speed (km/h)')
plt.ylabel('Frequency')
plt.show()
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
