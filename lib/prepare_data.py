import argparse
import os
import sys
import pandas as pd
import numpy as np
import re
from datetime import datetime

def parse_log_file(file_path):
    """Parse a single log file and return a dataframe"""
    try:
        node_id = os.path.basename(file_path).split('.')[0]
        #print(f"Processing {node_id}")
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 3:
            print(f"  File too short: {len(lines)} lines")
            return None
            
        # Extract header line
        header_line = lines[0].strip()
        header_parts = re.split(r'\s+', header_line)[2:]  # Skip timestamp and '#Entity'
        headers = ['timestamp', 'node_id', 'gpu_id'] + header_parts
        
        # Skip units line (line 1)
        
        # Process data lines (starting from line 2)
        data = []
        
        for i in range(2, len(lines)):
            line = lines[i].strip()
            if not line:
                continue
                
            # Check if this is a GPU line
            if "GPU" not in line:
                continue
                
            parts = line.split(None, 2)  # Split by whitespace, max 2 splits
            if len(parts) < 3:
                continue
                
            timestamp_str = parts[0]
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%f')
            except ValueError:
                continue
                
            # For GPU lines, parts looks like:
            # ["timestamp", "GPU", "ID  metrics..."]
            remaining = parts[2].split()
            
            # First element should be the GPU ID
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
            print(f"  No valid data rows found in {file_path}")
            return None
            
        #print(f"  Successfully parsed {len(data)} rows from {file_path}")
        
        # Create dataframe
        df = pd.DataFrame(data, columns=headers)

        # Convert Timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Round timestamp down to 0.1 seconds
        df['timestamp'] = df['timestamp'].dt.floor('100ms')
        # Set the index to the timestamp
        df.set_index('timestamp', inplace=True)
        
        # Convert numeric columns to float
        for col in df.columns[3:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


def create_df_dict(log_folder):
    """ 
    Create a dictionary with a dataframe for each node.
    The log files are expected to be in the format nid00XXXX.log, where XXXX is the node number.
    The function will process each log file and store the resulting dataframe in a dictionary.
    The keys of the dictionary will be the node numbers (last 4 digits of the file name before the .log).
    The values will be the dataframes.
    """
    # create a dictionary with a dataframe for each node
    log_dict = {}

    min_time = None
    min_node = None
    max_time = None
    max_node = None
    # loop over all files in the folder
    for file_name in os.listdir(log_folder):
        if file_name.endswith(".log"):
            # create the full path to the file
            file_path = os.path.join(log_folder, file_name)
            # get the number of the node (last 4 digits of the file name before the .log)
            node_number = file_name.split(".")[0][-4:]
            # process the log file and store the dataframe in the dictionary
            #log_dict[node_number] = parse_log_to_dataframe(file_path)
            log_dict[node_number] = parse_log_file(file_path)

    return log_dict, min_time, max_time

def process_all_logs(dir, limit=None):
    """Process all log files from a directory and return a combined dataframe.
    
    Args:
        dir: Directory containing the log files
        limit: If specified, limit the number of files to process from each run
    """

    # Check if directories exist
    if not os.path.exists(dir):
        print(f"Warning: Fast run directory does not exist: {dir}")
    
    # Process fast run logs
    run_files = [os.path.join(dir, file_name) for file_name in os.listdir(dir) if file_name.endswith(".log")]
    print(f"Found {len(run_files)} log files")
    
    if limit is not None and limit > 0:
        print(f"Limiting to {limit} files per run")
        run_files = run_files[:limit]
    
    dfs = []
    for file_path in run_files:
        df = parse_log_file(file_path)
        if df is not None:
            dfs.append(df)

    # Combine all dataframes into one
    combined_df = pd.concat(dfs)
    
    # Sort by timestamp
    combined_df = combined_df.sort_values(by=['timestamp', 'node_id', 'gpu_id'])
    
    return combined_df


def main():
    parser = argparse.ArgumentParser(description="Process SLURM job data.")
    parser.add_argument("--job-id", required=True, help="SLURM Job ID")
    args = parser.parse_args()

    # get the current date in the format DAY-MONTH-YEAR
    current_date = datetime.now().strftime("%d-%m-%Y")

    # prepare the paths used in the script
    logs_directory = f"./outputs/{args.job_id}_{current_date}/logs"
    output_file = f"./outputs/{args.job_id}_{current_date}/data_{args.job_id}_{current_date}"

    if not os.path.exists(logs_directory):
        print(f"Directory {logs_directory} does not exist.")
        sys.exit(1)

    combined_df = process_all_logs(logs_directory)

    if combined_df is not None:
        # Save to CSV
        output_csv = f"{output_file}.csv"
        combined_df.to_csv(output_csv)
        print(f"Data saved to {output_csv}")
        
        # Save to parquet (more efficient for large datasets)
        output_parquet = f"{output_file}.parquet"
        combined_df.to_parquet(output_parquet)
        print(f"Data saved to {output_parquet}")



if __name__ == "__main__":
    main()