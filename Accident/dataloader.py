from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import argparse
import numpy as np
import pandas as pd

def generate_graph_seq2seq_io_data(
    df, x_offsets, y_offsets, add_time_in_day=False, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """
    # Ensure the datetime columns are in datetime type

    print(f'df shape: {df.shape}')

    num_samples, num_nodes = df.shape
    print(f"Number of samples: {num_samples}, Number of nodes: {num_nodes}")
    data = np.expand_dims(df, axis=-1)  # This will add an extra dimension for features
    print(f"Expanded data shape: {data.shape}")
    feature_list = [data]

    # Add the time of day feature if required
    if add_time_in_day:
        time_ind = (df['Trip Start Timestamp'].dt.hour + df['Trip Start Timestamp'].dt.minute / 60.0)
        time_in_day = np.tile(time_ind.values, [1, 1]).T
        feature_list.append(time_in_day)

    # Add the day of the week feature if required
    if add_day_in_week:
        dow = df['Trip Start Timestamp'].dt.dayofweek
        dow_tiled = np.tile(dow.values, [1, 1]).T
        feature_list.append(dow_tiled)

    # Concatenate all features along the last axis
    data = np.concatenate(feature_list, axis=-1)

    # Generate the input and output sequences
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y

def generate_train_val_test(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    column_names = ['Trip Start Timestamp', 'Trip End Timestamp', 'Trip Seconds', 'Trip Miles',
                    'Pickup Community Area']
    df = pd.read_csv(args.traffic_df_filename, usecols=column_names)

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
    hourly_speeds = df.groupby(['Pickup Community Area', 'Trip Start Date', 'Trip Start Hour'])[
        'Speed'].mean().reset_index()
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
    # 转换为NumPy数组
    df_array = speed_df.values

    print(f'df_array: {df_array.shape}')

    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
    x_offsets = np.nan_to_num(x_offsets)

    # 替换 y 中的 NaN 为 0
    y_offsets = np.nan_to_num(y_offsets)
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df_array,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=False,
        add_day_in_week=args.dow,
    )


    x = x.astype(float)
    y = y.astype(float)

    # Write the data into npz file.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)


    num_val = num_samples - num_test - num_train
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]
    # Ensure the output directory exists
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        # Print the file path where the data will be saved
        file_path = os.path.join(output_dir, f"{cat}.npz")
        print(f"Attempting to save to: {file_path}")

        try:
            np.savez_compressed(file_path, x=_x, y=_y)
            print(f"Successfully saved {cat} data to {file_path}")
        except Exception as e:
            print(f"Failed to save {cat} data: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/temporal_data/Chicago", help="Output directory.")
    parser.add_argument("--traffic_df_filename", type=str, default="data/Chicago/2016_traffic/Taxi_Trips.csv", help="Raw traffic readings.",)
    parser.add_argument("--seq_length_x", type=int, default=24, help="Sequence Length.",)
    parser.add_argument("--seq_length_y", type=int, default=24, help="Sequence Length.",)
    parser.add_argument("--y_start", type=int, default=1, help="Y pred start", )
    parser.add_argument("--dow", action='store_true',)

    args = parser.parse_args()

    generate_train_val_test(args)