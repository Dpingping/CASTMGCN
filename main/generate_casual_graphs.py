import sys
import time

import pandas as pd
import numpy as np
import torch
# Add model directory to Python path
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../model_dynotears')))
# sys.path('../model')
import pickle
from multi_lag_dynotears import *

from sklearn.preprocessing import StandardScaler
import argparse


# Generate causal graphs by calendar week
def process_weekly_data(data, save_path='../../data/METR-LA/causalgraphs/', P=2, lambda1=0.001, lambda2=0.001, max_iter=100, device=None, start=None, end=None):
    """
    Generate weekly causal graphs using Dynotears (multi-lag version).

    :param data: Input data as a DataFrame with a datetime index
    :param save_path: Directory to save results
    :param P: Temporal lag order
    :param lambda1: L1 regularization coefficient
    :param lambda2: L2 regularization coefficient
    :param max_iter: Maximum number of optimization iterations
    :param device: torch.device or None for auto selection
    :param start: Optional inclusive start date string, e.g., '2012-03-01'
    :param end: Optional inclusive end date string, e.g., '2012-04-01'
    """
    # Select device
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

    # Filter by time range if provided
    data.index = pd.to_datetime(data.index)
    if start is not None:
        start = pd.to_datetime(start)
        data = data.loc[start:]
    if end is not None:
        end = pd.to_datetime(end)
        data = data.loc[:end]

    # Determine the first and last timestamps
    start_date = data.index.min()
    end_date = data.index.max()

    scaler = StandardScaler()

    # Find the first Monday as the starting point
    while start_date.weekday() != 0:
        start_date += pd.Timedelta(days=1)
    start_date = pd.Timestamp(start_date.date())

    current_date = start_date
    while current_date <= end_date:
        week_end = current_date + pd.Timedelta(days=7)
        week_data = data.loc[current_date:week_end - pd.Timedelta(seconds=1)]

        if not week_data.empty:
            week_str = current_date.strftime('%Y%m%d')
            print(f"\nProcessing data from {current_date.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}")

            # Prepare data
            normalized_data = scaler.fit_transform(week_data.values)
            X_tensor = torch.tensor(normalized_data, dtype=torch.float32)

            start_time = time.time()

            P_est_list, Z_t = dynotears_model(
                Xlags=X_tensor,
                P=P,
                device=device,
                lambda1=lambda1,
                lambda2=lambda2,
                max_iter=max_iter
            )
            save_dir = save_path
            os.makedirs(save_dir, exist_ok=True)
            file_path = os.path.join(save_dir, f'week_{week_str}.npz')
            np.savez(file_path,
                     P_est_list=P_est_list,
                     Z_t=Z_t)

            end_time = time.time()
            print(f"Elapsed time: {end_time - start_time:.2f} seconds")

        # Move to next Monday
        current_date = week_end


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate weekly causal graphs using multi-lag Dynotears')
    parser.add_argument('--traffic_df_filename', type=str, default='../data/HK/data_hk.csv', help='Input CSV path (first column is datetime index)')
    parser.add_argument('--output_dir', type=str, default='../data/HK/causalgraphs/', help='Output directory for weekly npz files')
    parser.add_argument('--p', type=int, default=2, help='Lag order P')
    parser.add_argument('--lambda1', type=float, default=0.001, help='L1 regularization coefficient')
    parser.add_argument('--lambda2', type=float, default=0.001, help='L2 regularization coefficient')
    parser.add_argument('--max_iter', type=int, default=100, help='Maximum iterations')
    parser.add_argument('--device', type=str, default='cuda', choices=['auto', 'cuda', 'mps', 'cpu'], help='Computation device')
    parser.add_argument('--start', type=str, default=None, help='Start date (YYYY-MM-DD, inclusive, optional)')
    parser.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD, inclusive, optional)')

    args = parser.parse_args()

    # Resolve device
    if args.device == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    elif args.device == 'cuda':
        device = torch.device('cuda:0')
    elif args.device == 'mps':
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Load data
    data = pd.read_csv(args.traffic_df_filename, index_col=0, parse_dates=True)
    print("data.head():\n", data.head())

    # Execute
    process_weekly_data(
        data,
        save_path=args.output_dir,
        P=args.p,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        max_iter=args.max_iter,
        device=device,
        start=args.start,
        end=args.end
    )
