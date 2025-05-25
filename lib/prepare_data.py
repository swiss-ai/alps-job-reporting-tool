import argparse
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd


def parse_gpu_file(file_path):
    """Parse a single log file and return a dataframe"""
    try:
        node_id = os.path.basename(file_path).split('_')[0]

        with open(file_path, 'r') as f:
            lines = f.readlines()

        if len(lines) < 3:
            print(f'  File too short: {len(lines)} lines')
            return None

        # Extract header line
        header_line = lines[0].strip()
        header_parts = re.split(r'\s+', header_line)[2:]  # Skip timestamp and '#Entity'
        headers = ['timestamp', 'node_id', 'gpu_id'] + header_parts

        # Process data lines (starting from line 2)
        data = []

        for i in range(2, len(lines)):
            line = lines[i].strip()

            if not line:
                continue

            # Check if this is a GPU line
            if 'GPU' not in line:
                continue

            parts = line.split(None, 2)  # Split by whitespace, max 2 splits
            if len(parts) < 3:
                continue

            timestamp_str = parts[0]
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%fZ')

            except ValueError as e:
                print(f'Error parsing timestamp: {e}')

            # For GPU lines, parts looks like:
            # ['timestamp', 'GPU', 'ID  metrics...']
            remaining = parts[2].split()

            # The first element should be the GPU ID
            if len(remaining) < 1:
                continue

            gpu_id = remaining[0]
            metrics = remaining[1:]

            # Replace N/A with np.nan
            metrics = [np.nan if x == 'N/A' else x for x in metrics]

            # Create a row
            row = [timestamp, node_id, gpu_id] + metrics
            data.append(row)

        if not data:
            print(f'  No valid data rows found in {file_path}')
            return None

        # Create dataframe
        df = pd.DataFrame(data, columns=headers)

        # Convert Timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Round timestamp down to 0.1 seconds
        df['timestamp'] = df['timestamp'].dt.floor('100ms')
        # Convert timestamp to datetime in the same format as other DataFrames
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)

        # Convert numeric columns to float
        for col in df.columns[3:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    except Exception as e:
        print(f'Error processing file {file_path}: {e}')
        return None


def process_gpu_logs(dir, limit=None):
    """Process all log files from a directory and return a combined dataframe.
    
    Args:
        dir: Directory containing the GPU log files
        limit: If specified, limit the number of files to process from each run
    """

    # Check if directories exist
    if not os.path.exists(dir):
        print(f'Warning: Fast run directory does not exist: {dir}')

    # Process fast run logs
    run_files = [os.path.join(dir, file_name) for file_name in os.listdir(dir) if file_name.endswith('gpu.log')]
    print(f'Found {len(run_files)} GPU log files')

    if limit is not None and limit > 0:
        print(f'Limiting to {limit} files per run')
        run_files = run_files[:limit]

    dfs = []
    for file_path in run_files:
        df = parse_gpu_file(file_path)
        if df is not None:
            dfs.append(df)

    # Combine all dataframes into one
    combined_df = pd.concat(dfs)

    # Sort by timestamp
    combined_df = combined_df.sort_values(by=['timestamp', 'node_id', 'gpu_id'])

    return combined_df


def parse_cpu_file(file_path):
    """Parse a single CPU log file and return a DataFrame."""
    try:
        node_id = os.path.basename(file_path).split('_')[0]  # Extract node ID from file name

        # Read the log file into a DataFrame
        df = pd.read_csv(file_path)

        # strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Add the node_id column
        df['node_id'] = node_id

        # Extract the type of metric (e.g., temp, power, curr) from the sensor column
        df['metric_type'] = df['sensor'].str.extract(r'(temp|power|curr)')

        # Extract the unit (e.g., °C, W, A) from the value column
        df['value'] = df['value'].str.extract(r'([\d.]+)').astype(float)

        return df

    except Exception as e:
        print(f'Error processing file {file_path}: {e}')
        return None


def process_cpu_logs(dir, limit=None):
    """Process all log files from a directory and return a combined dataframe.
    
    Args:
        dir: Directory containing the CPU temp/power/curr log files
        limit: If specified, limit the number of files to process from each run
    """

    # Check if directories exist
    if not os.path.exists(dir):
        print(f'Warning: Fast run directory does not exist: {dir}')

    # Process fast run logs
    run_files = [os.path.join(dir, file_name) for file_name in os.listdir(dir) if file_name.endswith('cpu.log')]
    print(f'Found {len(run_files)} CPU log files')

    if limit is not None and limit > 0:
        print(f'Limiting to {limit} files per run')
        run_files = run_files[:limit]

    dfs = []
    for file_path in run_files:
        df = parse_cpu_file(file_path)
        if df is not None:
            # Group the data by timestamp, node_id and metric_type and aggregate the values (min, max, median, mean)
            df = df.groupby(['timestamp', 'node_id', 'metric_type']) \
                .agg({'value': ['min', 'max', 'median', 'mean']}).reset_index()
            dfs.append(df)

    # Combine all dataframes into one
    combined_df = pd.concat(dfs)

    # Convert timestamp to datetime
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], unit='s', utc=True)

    # Pivot the table to have one row per timestamp and node_id
    combined_df.columns = ['timestamp', 'node_id', 'metric_type', 'min', 'max', 'median', 'mean']
    pivot_df = combined_df.pivot_table(
        index=['timestamp', 'node_id'],
        columns='metric_type',
        values=['min', 'max', 'median', 'mean']
    )

    # Flatten the multi-level columns
    pivot_df.columns = [f'{metric_type}_{agg}' for agg, metric_type in pivot_df.columns]
    pivot_df = pivot_df.reset_index()

    # Sort by timestamp and node_id
    pivot_df = pivot_df.sort_values(by=['timestamp', 'node_id'])

    return pivot_df


def parse_cpu_util_file(file_path):
    """
    Parse a single CPU utilization log file and return a DataFrame.

    Args:
        file_path (str): Path to the CPU utilization log file.

    Returns:
        pd.DataFrame: Parsed DataFrame with CPU utilization data.
    """
    try:
        # Read the log file into a DataFrame
        df = pd.read_csv(file_path)

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Convert the timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)

        # Add a node_id column based on the file name
        node_id = os.path.basename(file_path).split('_')[0]
        df['node_id'] = node_id

        return df
    except Exception as e:
        print(f'Error processing file {file_path}: {e}')
        return None


def process_cpu_util_logs(directory, limit=None):
    """
    Process all CPU utilization log files in a directory and return a combined DataFrame
    with differences in utilization metrics compared to the previous timestamp.

    Args:
        directory (str): Directory containing the CPU utilization log files.
        limit (int, optional): Limit the number of files to process. Defaults to None.

    Returns:
        pd.DataFrame: Combined DataFrame with utilization differences.
    """
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f'Directory does not exist: {directory}')
        return None

    # Find all CPU utilization log files
    cpu_util_files = [
        os.path.join(directory, file_name)
        for file_name in os.listdir(directory)
        if file_name.endswith('cpu_util.log')
    ]
    print(f'Found {len(cpu_util_files)} CPU utilization log files.')

    if limit is not None and limit > 0:
        print(f'Limiting to {limit} files.')
        cpu_util_files = cpu_util_files[:limit]

    # Parse each file and calculate differences
    dfs = []
    for file_path in cpu_util_files:
        df = parse_cpu_util_file(file_path)
        if df is not None:
            # Calculate differences for utilization metrics
            utilization_columns = [
                'user', 'nice', 'system', 'idle', 'iowait', 'irq', 'softirq', 'steal', 'guest', 'guest_nice',
            ]
            df[utilization_columns] = df[utilization_columns].diff()

            # Drop the first row for each node (since diff() produces NaN for the first row)
            df = df.dropna(subset=utilization_columns)

            dfs.append(df)

    # Combine all DataFrames into one
    combined_df = pd.concat(dfs, ignore_index=True)

    # Sort by timestamp and node_id
    combined_df = combined_df.sort_values(by=['timestamp', 'node_id'])

    # Reset the index after sorting
    combined_df = combined_df.reset_index(drop=True)

    return combined_df


def parse_io_file(file_path):
    """
    Parse a single I/O log file and return a DataFrame.

    Args:
        file_path (str): Path to the I/O log file.

    Returns:
        pd.DataFrame: Parsed DataFrame with I/O data.
    """
    try:
        # Read the log file into a DataFrame
        df = pd.read_csv(file_path)

        # strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Convert the timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)

        # Add a node_id column based on the file name
        node_id = os.path.basename(file_path).split('_')[0]
        df['node_id'] = node_id

        return df
    except Exception as e:
        print(f'Error processing file {file_path}: {e}')
        return None


def process_io_logs(directory, limit=None):
    """
    Process all I/O log files in a directory and return a combined DataFrame
    with differences in I/O metrics compared to the previous timestamp.

    Args:
        directory (str): Directory containing the I/O log files.
        limit (int, optional): Limit the number of files to process. Defaults to None.

    Returns:
        pd.DataFrame: Combined DataFrame with I/O differences.
    """
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f'Directory does not exist: {directory}')
        return None

    # Find all I/O log files
    io_files = [
        os.path.join(directory, file_name)
        for file_name in os.listdir(directory)
        if file_name.endswith('io.log')
    ]
    print(f'Found {len(io_files)} I/O log files.')

    if limit is not None and limit > 0:
        print(f'Limiting to {limit} files.')
        io_files = io_files[:limit]

    # Parse each file and calculate differences
    dfs = []
    for file_path in io_files:
        df = parse_io_file(file_path)
        if df is not None:
            # Calculate differences for I/O metrics
            io_columns = ['rchar', 'wchar', 'syscr', 'syscw', 'read_bytes', 'write_bytes', 'cancelled_write_bytes']
            df[io_columns] = df[io_columns].diff()

            # Drop the first row for each node (since diff() produces NaN for the first row)
            df = df.dropna(subset=io_columns)

            dfs.append(df)

    # Combine all DataFrames into one
    combined_df = pd.concat(dfs, ignore_index=True)

    # Sort by timestamp and node_id
    combined_df = combined_df.sort_values(by=['timestamp', 'node_id'])

    # Reset the index after sorting
    combined_df = combined_df.reset_index(drop=True)

    return combined_df


def parse_net_file(file_path):
    """
    Parse a single network log file and return a DataFrame.

    Args:
        file_path (str): Path to the network log file.

    Returns:
        pd.DataFrame: Parsed DataFrame with network data.
    """
    try:
        # Read the log file into a DataFrame
        df = pd.read_csv(file_path)

        # strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Convert the timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)

        # Add a node_id column based on the file name
        node_id = os.path.basename(file_path).split('_')[0]
        df['node_id'] = node_id

        return df
    except Exception as e:
        print(f'Error processing file {file_path}: {e}')
        return None


def process_net_logs(directory, limit=None):
    """
    Process all network log files in a directory and return a combined DataFrame
    with separate columns for each interface.

    Args:
        directory (str): Directory containing the network log files.
        limit (int, optional): Limit the number of files to process. Defaults to None.

    Returns:
        pd.DataFrame: Combined DataFrame with pivoted network data.
    """
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f'Directory does not exist: {directory}')
        return None

    # Find all network log files
    net_files = [
        os.path.join(directory, file_name)
        for file_name in os.listdir(directory)
        if file_name.endswith('net.log')
    ]
    print(f'Found {len(net_files)} network log files.')

    if limit is not None and limit > 0:
        print(f'Limiting to {limit} files.')
        net_files = net_files[:limit]

    # Parse each file and collect DataFrames
    dfs = []
    for file_path in net_files:
        df = parse_net_file(file_path)
        if df is not None:
            dfs.append(df)

    # Combine all DataFrames into one
    combined_df = pd.concat(dfs, ignore_index=True)

    # Pivot the table to have separate columns for each interface
    pivot_df = combined_df.pivot_table(
        index=['timestamp', 'node_id'],
        columns='interface',
        values=['rx_bytes', 'rx_packets', 'rx_errors', 'tx_bytes', 'tx_packets', 'tx_errors']
    )

    # Flatten the multi-level columns
    pivot_df.columns = [f'{interface}_{metric}' for metric, interface in pivot_df.columns]
    pivot_df = pivot_df.reset_index()

    # Sort by timestamp and node_id
    pivot_df = pivot_df.sort_values(by=['timestamp', 'node_id'])

    # Calculate differences for each metric
    diff_columns = pivot_df.columns.difference(['timestamp', 'node_id'])  # Exclude index columns
    pivot_df[diff_columns] = pivot_df.groupby('node_id')[diff_columns].diff()

    return pivot_df


def main():
    parser = argparse.ArgumentParser(description='Process SLURM job data.')
    parser.add_argument('--job-id', required=True, help='SLURM Job ID')
    args = parser.parse_args()

    # Get the current date in the format DAY-MONTH-YEAR
    current_date = datetime.now().strftime('%d-%m-%Y')

    # Prepare the paths used in the script
    logs_directory = f'./outputs/{args.job_id}_{current_date}/logs'
    output_file = f'./outputs/{args.job_id}_{current_date}/data_{args.job_id}_{current_date}'

    if not os.path.exists(logs_directory):
        print(f'Directory {logs_directory} does not exist.')
        sys.exit(1)

    gpu_df = process_gpu_logs(logs_directory)
    cpu_df = process_cpu_logs(logs_directory)
    cpu_util_df = process_cpu_util_logs(logs_directory)
    io_df = process_io_logs(logs_directory)
    net_df = process_net_logs(logs_directory)

    # Merge the DataFrames on timestamp and node_id
    combined_df = pd.merge(cpu_df, cpu_util_df, on=['timestamp', 'node_id'], how='outer')
    combined_df = pd.merge(combined_df, io_df, on=['timestamp', 'node_id'], how='outer')
    combined_df = pd.merge(combined_df, net_df, on=['timestamp', 'node_id'], how='outer')

    # Reset the index after merging
    combined_df = combined_df.reset_index(drop=True)

    # Strip whitespace from column names
    combined_df.columns = combined_df.columns.str.strip()

    print(f'Number of rows with at least one NaN value: {combined_df.isnull().any(axis=1).sum()}')

    if combined_df is not None:
        # Save to CSV the combined DataFrame
        output_csv = f'{output_file}.csv'
        combined_df.to_csv(output_csv)
        print(f'Node data saved to {output_csv}')

        # Save to parquet (more efficient for large datasets)
        output_parquet = f'{output_file}.parquet'
        combined_df.to_parquet(output_parquet)
        print(f'Node data saved to {output_parquet}')

        # Save the GPU data to CSV
        gpu_output_csv = f'{output_file}_gpu.csv'
        gpu_df.to_csv(gpu_output_csv)
        print(f'GPU data saved to {gpu_output_csv}')

        # Save the GPU data to Parquet
        gpu_output_parquet = f'{output_file}_gpu.parquet'
        gpu_df.to_parquet(gpu_output_parquet)
        print(f'GPU data saved to {gpu_output_parquet}')


if __name__ == '__main__':
    main()
