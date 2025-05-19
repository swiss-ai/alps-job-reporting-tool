#!/bin/bash

# filepath: /users/ndegiorgi/store/alps-job-reporting-tool/lib/save_cpu_util.sh

# Set base directory for logs
BASE_DIR="."

# Retrieve SLURM job ID and node name
JOB_ID="${SLURM_JOB_ID:-unknown_job}"
NODE_NAME="${SLURMD_NODENAME:-$(hostname)}"

# Get the current date in the format DAY-MONTH-YEAR
CURRENT_DATE=$(date +%d-%m-%Y)

# Define output directory and file
OUTPUT_DIR="${BASE_DIR}/outputs/${JOB_ID}_${CURRENT_DATE}/logs"
CPU_UTIL_OUTPUT_FILE="${OUTPUT_DIR}/${NODE_NAME}_cpu_util.log"

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
echo "timestamp, user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice" > "$CPU_UTIL_OUTPUT_FILE"

# Initialize the previous timestamp
prev_timestamp=$(date +%s)

# Loop to log data whenever the timestamp increases
counter=0
while [[ $counter -lt $DURATION ]]; do
  current_timestamp=$(date +%s)
  if [[ $current_timestamp -ne $prev_timestamp ]]; then
    prev_timestamp=$current_timestamp
    timestamp=$current_timestamp

    # Read CPU utilization data from /proc/stat
    cpu_line=$(grep '^cpu ' /proc/stat)
    user=$(echo "$cpu_line" | awk '{print $2}')
    nice=$(echo "$cpu_line" | awk '{print $3}')
    system=$(echo "$cpu_line" | awk '{print $4}')
    idle=$(echo "$cpu_line" | awk '{print $5}')
    iowait=$(echo "$cpu_line" | awk '{print $6}')
    irq=$(echo "$cpu_line" | awk '{print $7}')
    softirq=$(echo "$cpu_line" | awk '{print $8}')
    steal=$(echo "$cpu_line" | awk '{print $9}')
    guest=$(echo "$cpu_line" | awk '{print $10}')
    guest_nice=$(echo "$cpu_line" | awk '{print $11}')

    # Write the metrics to the log file
    echo "$timestamp, $user, $nice, $system, $idle, $iowait, $irq, $softirq, $steal, $guest, $guest_nice" >> "$CPU_UTIL_OUTPUT_FILE"

    counter=$((counter + 1))
  fi
  sleep 0.1
done

#echo "CPU utilization logging complete. Output saved to ${CPU_UTIL_OUTPUT_FILE}"