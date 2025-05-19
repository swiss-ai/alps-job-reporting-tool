#!/bin/bash

# filepath: /users/ndegiorgi/store/alps-job-reporting-tool/lib/save_net_metrics.sh
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
NET_OUTPUT_FILE="${OUTPUT_DIR}/${NODE_NAME}_net.log"

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
echo "timestamp, interface, rx_bytes, rx_packets, rx_errors, tx_bytes, tx_packets, tx_errors" > "$NET_OUTPUT_FILE"

# Initialize the previous timestamp
prev_timestamp=$(date +%s)

# Loop to log data whenever the timestamp increases
counter=0
while [[ $counter -lt $DURATION ]]; do
  current_timestamp=$(date +%s)
  if [[ $current_timestamp -ne $prev_timestamp ]]; then
    prev_timestamp=$current_timestamp
    timestamp=$current_timestamp

    # Read network metrics from /proc/net/dev
    while read -r line; do
      # Extract interface name and metrics
      if [[ $line =~ ^\ *(hsn[0-9]+|nmn0): ]]; then
        interface=$(echo "$line" | awk -F: '{print $1}' | xargs)
        rx_bytes=$(echo "$line" | awk '{print $2}')
        rx_packets=$(echo "$line" | awk '{print $3}')
        rx_errors=$(echo "$line" | awk '{print $4}')
        tx_bytes=$(echo "$line" | awk '{print $10}')
        tx_packets=$(echo "$line" | awk '{print $11}')
        tx_errors=$(echo "$line" | awk '{print $12}')
        echo "$timestamp, $interface, $rx_bytes, $rx_packets, $rx_errors, $tx_bytes, $tx_packets, $tx_errors" >> "$NET_OUTPUT_FILE"
      fi
    done < /proc/net/dev

    counter=$((counter + 1))
  fi
  sleep 0.1
done

#echo "Network metrics logging complete. Output saved to ${NET_OUTPUT_FILE}"