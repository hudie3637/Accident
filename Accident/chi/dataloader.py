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
    result_df = expanded_df.groupby(
        ['Time Point Date', 'Time Point Hour', 'Time Point Minute', 'Pickup Community Area']).agg(
        {'Speed': 'mean'}).reset_index()

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