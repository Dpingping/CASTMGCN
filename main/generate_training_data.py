from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
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
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    feature_list = [data]

    # Add time features
    df.index = pd.to_datetime(df.index)

    # Add timestamps (Unix timestamp, seconds)
    timestamps = df.index.astype(np.int64) // 10 ** 9
    timestamps = np.tile(timestamps, [1, num_nodes, 1]).transpose((2, 1, 0))
    feature_list.append(timestamps)

    if add_time_in_day:
        # Calculate time of day (decimal between 0-1)
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)

    if add_day_in_week:
        # Add day of week (0-6)
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    data = np.concatenate(feature_list, axis=-1)
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

    # Read data
    df = pd.read_csv(args.traffic_df_filename, index_col=0, parse_dates=True)
    print("Data shape:", df.shape)
    print("Data sample:")
    print(df.head())
    print("\nTime range:", df.index.min(), "to", df.index.max())

    # 0 is the latest observed sample.
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))

    # Generate training data
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=args.dow,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)

    # Split into train, validation and test sets (configurable ratios)
    num_samples = x.shape[0]
    train_r = float(args.train_ratio)
    val_r = float(args.val_ratio)
    test_r = float(args.test_ratio)
    total_r = train_r + val_r + test_r
    # Normalize if sum is not 1 due to input or floating error
    if not (0.999 <= total_r <= 1.001):
        train_r, val_r, test_r = (r / total_r for r in (train_r, val_r, test_r))

    num_train = round(num_samples * train_r)
    num_test = round(num_samples * test_r)
    num_val = num_samples - num_test - num_train
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]

    # Save data
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}_data.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--output_dir", type=str, default="../../data/HK/", help="Output directory.")
    parser.add_argument("--output_dir", type=str, default="../data/HK/", help="Output directory.")
    parser.add_argument("--traffic_df_filename", type=str, default="../data/HK/data_hk.csv",
                        help="Raw traffic readings.", )
    parser.add_argument("--seq_length_x", type=int, default=12, help="Input sequence length.", )
    parser.add_argument("--seq_length_y", type=int, default=12, help="Output sequence length.", )
    parser.add_argument("--y_start", type=int, default=1, help="Y prediction start offset", )
    parser.add_argument("--dow", action='store_true', help="Add day of week feature")
    parser.add_argument("--train_ratio", type=float, default=0.6, help="Training split ratio (default: 0.6)")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio (default: 0.2)")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test split ratio (default: 0.2)")

    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        reply = str(input(f'{args.output_dir} exists. Do you want to overwrite it? (y/n)')).lower().strip()
        if reply[0] != 'y': exit
    else:
        os.makedirs(args.output_dir)
    generate_train_val_test(args)