#!/bin/bash

# filepath: /users/ndegiorgi/store/alps-job-reporting-tool/lib/save_io_metrics.sh

# Set base directory for logs
BASE_DIR="."

# Retrieve SLURM job ID and node name
JOB_ID="${SLURM_JOB_ID:-unknown_job}"
NODE_NAME="${SLURMD_NODENAME:-$(hostname)}"

# Get the current date in the format DAY-MONTH-YEAR
CURRENT_DATE=$(date +%d-%m-%Y)

# Define output directory and file
OUTPUT_DIR="${BASE_DIR}/outputs/${JOB_ID}_${CURRENT_DATE}/logs"
IO_OUTPUT_FILE="${OUTPUT_DIR}/${NODE_NAME}_io.log"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Get the duration of the intended logging session
DURATION=300 # Default duration in seconds

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --duration) DURATION="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Print the header to the log file
echo "timestamp, rchar, wchar, syscr, syscw, read_bytes, write_bytes, cancelled_write_bytes" > "$IO_OUTPUT_FILE"

# Initialize the previous timestamp
prev_timestamp=$(date +%s)

# Loop to log data whenever the timestamp increases
counter=0
while [[ $counter -lt $DURATION ]]; do
  current_timestamp=$(date +%s)
  if [[ $current_timestamp -ne $prev_timestamp ]]; then
    prev_timestamp=$current_timestamp
    timestamp=$current_timestamp

    # Read I/O metrics from /proc/self/io
    while read -r line; do
      case $line in
        rchar:*) rchar=$(echo "$line" | awk '{print $2}') ;;
        wchar:*) wchar=$(echo "$line" | awk '{print $2}') ;;
        syscr:*) syscr=$(echo "$line" | awk '{print $2}') ;;
        syscw:*) syscw=$(echo "$line" | awk '{print $2}') ;;
        read_bytes:*) read_bytes=$(echo "$line" | awk '{print $2}') ;;
        write_bytes:*) write_bytes=$(echo "$line" | awk '{print $2}') ;;
        cancelled_write_bytes:*) cancelled_write_bytes=$(echo "$line" | awk '{print $2}') ;;
      esac
    done < /proc/self/io

    # Write the metrics to the log file
    echo "$timestamp, $rchar, $wchar, $syscr, $syscw, $read_bytes, $write_bytes, $cancelled_write_bytes" >> "$IO_OUTPUT_FILE"

    counter=$((counter + 1))
  fi
  sleep 0.1
done

#echo "I/O metrics logging complete. Output saved to ${IO_OUTPUT_FILE}"