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
    df['Trip Start Timestamp'] = pd.to_datetime(df['Trip Start Timestamp'])
    df['Trip End Timestamp'] = pd.to_datetime(df['Trip End Timestamp'])

    # 然后转换为UNIX时间戳（浮点数）
    df['Trip Start Timestamp'] = (df['Trip Start Timestamp'] - pd.Timestamp('1970-01-01 00:00:00')) / np.timedelta64(1,
                                                                                                                     's')
    df['Trip End Timestamp'] = (df['Trip End Timestamp'] - pd.Timestamp('1970-01-01 00:00:00')) / np.timedelta64(1, 's')

    print(df.info())

    # Initialize the list of features
    feature_list = []

    # Convert the DataFrame to a numpy array, adding an extra dimension
    data = np.expand_dims(df.values, axis=-1)
    feature_list.append(data)

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
    max_t = abs(len(df) - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, :, :])
        y.append(data[t + y_offsets, :, :])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    print("Data type of elements in x:", x.dtype)
    print("Data type of elements in y:", y.dtype)

    return x, y

def generate_train_val_test(args):


    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y

    df = pd.read_csv(args.traffic_df_filename)
    # 将 'Trip Start Timestamp' 和 'Trip End Timestamp' 列转换为 Unix 时间戳，并保留为浮点数类型

    # 0 is the latest observed sample.
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=False,
        add_day_in_week=args.dow,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
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
