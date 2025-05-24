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
GPU_OUTPUT_FILE="${OUTPUT_DIR}/${NODE_NAME}_gpu.log"
CPU_OUTPUT_FILE="${OUTPUT_DIR}/${NODE_NAME}_cpu.log"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Define metrics to monitor (e.g., GPU utilization, memory utilization, power usage, temperature)
METRICS="140,150,155,157,190,191,200,201,203,204,206,207,254,504,858,861,862,1001,1002,1003,1004,1005,1009,1010,1011,1012,1040,1041,1042,1043,1044,1045,1046,1047"

# Duration in seconds
DURATION=300

# Sampling interval in milliseconds
INTERVAL=100

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --duration) DURATION="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Calculate the number of samples
SAMPLES=$((DURATION * 1000 / INTERVAL))

# Start monitoring
dcgmi dmon -e "${METRICS}" -d "${INTERVAL}" -c "${SAMPLES}" | while read -r line; do
    echo "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ) ${line}"
done > "${GPU_OUTPUT_FILE}"

#echo GPU monitoring complete. Output saved to ${OUTPUT_FILE}"
