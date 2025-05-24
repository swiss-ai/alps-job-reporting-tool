#!/bin/bash

# ------------------------------------------------------------------
# Run save_snapshot.sh on every node of an existing SLURM job.
# Each node executes the script in parallel using srun --overlap.
#
# Usage:
#   ./run_save_snapshot_on_nodes.sh <job_id> [duration]
#
# Requirements:
#   - The job must already be running.
#   - The script save_snapshot.sh must be readable on all nodes.
# ------------------------------------------------------------------

# ====== CONFIGURATION ======
CURRENT_DATE=$(date +%d-%m-%Y)

GPU_SNAPSHOT_SCRIPT="./lib/save_gpu_metrics.sh"
CPU1_SNAPSHOT_SCRIPT="./lib/save_cpu_temp.sh"
CPU2_SNAPSHOT_SCRIPT="./lib/save_cpu_util.sh"
NET_SNAPSHOT_SCRIPT="./lib/save_net_metrics.sh"
IO_SNAPSHOT_SCRIPT="./lib/save_io_metrics.sh"

PYTHON_SCRIPT="./lib/prepare_data.py"

REPORT_SCRIPT="./lib/report.py"
# ===========================

# Display help message
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Usage: $0 <job_id> [duration]"
    echo
    echo "Arguments:"
    echo "  <job_id>    The SLURM job ID for which the snapshot saver will run."
    echo "  [duration]  Optional. Duration in seconds for the snapshot saver to run. Default is 300 seconds."
    echo
    echo "Description:"
    echo "  This script performs the following steps:"
    echo "    1. Runs multiple data logging scripts (GPU, CPU, network, and I/O metrics) on all nodes of the specified SLURM job."
    echo "    2. Executes a Python script to prepare and merge the collected data into a unified format."
    echo "    3. Generates a web-based HTML report using the prepared data."
    echo
    echo "Steps in Detail:"
    echo "  - Data Logging: The following scripts are executed in parallel on all nodes:"
    echo "      - $GPU_SNAPSHOT_SCRIPT: Logs GPU metrics."
    echo "      - $CPU1_SNAPSHOT_SCRIPT: Logs CPU temperature metrics."
    echo "      - $CPU2_SNAPSHOT_SCRIPT: Logs CPU utilization metrics."
    echo "      - $NET_SNAPSHOT_SCRIPT: Logs network metrics."
    echo "      - $IO_SNAPSHOT_SCRIPT: Logs I/O metrics."
    echo
    echo "  - Data Preparation: The script $PYTHON_SCRIPT processes and merges the collected data."
    echo
    echo "  - Report Generation: The script $REPORT_SCRIPT creates an interactive HTML report."
    echo
    echo "Example:"
    echo "  $0 12345 600"
    echo "  This runs the data logging scripts for job ID 12345 with a duration of 600 seconds, prepares the data, and generates a web report."
    exit 0
fi

# Check for job ID argument
if [ -z "$1" ]; then
    echo "Usage: $0 <job_id>"
    exit 1
fi

JOB_ID="$1"

# If there is no duration argument, set a default of 300 seconds
DURATION="${2:-300}"

# Get list of nodes allocated to the job
NODELIST=$(squeue --job "$JOB_ID" --noheader --format="%R" | xargs scontrol show hostnames)

if [ -z "$NODELIST" ]; then
    echo "Error: Could not retrieve node list for job $JOB_ID."
    exit 1
fi

echo "Launching scripts on all nodes of job $JOB_ID..."

# NEW VERSION TO ADD LATER
# Launch one srun per node in the background
NODE_COUNT=0
for NODE in $NODELIST; do
    srun --overlap --jobid="$JOB_ID" --nodes=1 --ntasks=1 -w "$NODE" bash -c "
        bash \"$GPU_SNAPSHOT_SCRIPT\" --duration \"$DURATION\" &
        bash \"$CPU1_SNAPSHOT_SCRIPT\" --duration \"$DURATION\" &
        bash \"$CPU2_SNAPSHOT_SCRIPT\" --duration \"$DURATION\" &
        bash \"$NET_SNAPSHOT_SCRIPT\" --duration \"$DURATION\" &
        bash \"$IO_SNAPSHOT_SCRIPT\" --duration \"$DURATION\" &
        wait
    " &
    NODE_COUNT=$((NODE_COUNT + 1))
done

# Wait for all background sruns to finish
echo "Waiting for approximately $DURATION seconds..."
for ((i = 1; i <= DURATION; i++)); do
    # Calculate the number of '#' characters to display
    num_hashes=$((i * 20 / DURATION))  # Scale to a maximum of 20 '#' characters
    progress=$(printf "%-${num_hashes}s" "#" | tr ' ' '#')  # Create the progress bar
    printf "\rProgress: [%-20s] %d%%" "$progress" $((i * 100 / DURATION))
    sleep 1
done
printf "\n"


# Wait for all background sruns to finish
wait

echo "Finished data logging on $NODE_COUNT nodes."

# Start the Python script to prepare the data for analysis
echo "-------------"
echo "Starting Python script: $PYTHON_SCRIPT"
python3 "$PYTHON_SCRIPT" --job-id "$JOB_ID"

# file paths for the input and output files
INPUT_FILE="./outputs/${JOB_ID}_${CURRENT_DATE}/data_${JOB_ID}_${CURRENT_DATE}_gpu.parquet"
INPUT_FILE2="./outputs/${JOB_ID}_${CURRENT_DATE}/data_${JOB_ID}_${CURRENT_DATE}.parquet"
OUTPUT_FILE="./outputs/${JOB_ID}_${CURRENT_DATE}/report_${JOB_ID}.html"

# Start the reporting script once the Python script is done
echo "-------------"
echo "Generate report using following command:"
echo "python3 \"$REPORT_SCRIPT\" --input_file \"$INPUT_FILE\" --input_file2 \"$INPUT_FILE2\" --output_file \"$OUTPUT_FILE\""
python3 "$REPORT_SCRIPT" --input_file "$INPUT_FILE" --input_file2 "$INPUT_FILE2" --output_file "$OUTPUT_FILE"
