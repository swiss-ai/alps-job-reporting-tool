#!/bin/bash

# Set base directory for logs
BASE_DIR="."

# Retrieve SLURM job ID and node name
JOB_ID="${SLURM_JOB_ID:-unknown_job}"
NODE_NAME="${SLURMD_NODENAME:-$(hostname)}"

# Get the current date in the format DAY-MONTH-YEAR
CURRENT_DATE=$(date +%d-%m-%Y)

# Define output directory and file
OUTPUT_DIR="${BASE_DIR}/outputs/${JOB_ID}_${CURRENT_DATE}/logs"
CPU_OUTPUT_FILE="${OUTPUT_DIR}/${NODE_NAME}_cpu.log"

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
echo "timestamp, name, sensor, value" > "$CPU_OUTPUT_FILE"

# Initialize the previous timestamp
prev_timestamp=$(date +%s)

# Loop to log data whenever the timestamp increases
counter=0
while [[ $counter -lt $DURATION ]]; do
  current_timestamp=$(date +%s)
  if [[ $current_timestamp -ne $prev_timestamp ]]; then
    prev_timestamp=$current_timestamp
    timestamp=$current_timestamp
    for hwmon in /sys/class/hwmon/hwmon*; do
      name=$(cat "$hwmon/name")
      for temp in "$hwmon"/temp*_input; do
        [ -f "$temp" ] || continue
        value=$(cat "$temp")
        echo "$timestamp, $name, $(basename $temp), $((value / 1000)).$((value % 1000))°C" >> "$CPU_OUTPUT_FILE"
      done
      for power in "$hwmon"/power*_input; do
        [ -f "$power" ] || continue
        value=$(cat "$power")
        echo "$timestamp, $name, $(basename $power), $((value / 1000000)).$((value / 1000 % 1000)) W" >> "$CPU_OUTPUT_FILE"
      done
      for current in "$hwmon"/curr*_input; do
        [ -f "$current" ] || continue
        value=$(cat "$current")
        echo "$timestamp, $name, $(basename $current), $((value / 1000)).$((value % 1000)) A" >> "$CPU_OUTPUT_FILE"
      done
    done
    counter=$((counter + 1))
  fi
  sleep 0.1
done