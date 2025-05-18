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
GPU_SNAPSHOT_SCRIPT="./lib/save_gpu_metrics.sh"
CPU1_SNAPSHOT_SCRIPT="./lib/save_cpu_temp.sh"
CPU2_SNAPSHOT_SCRIPT="./lib/save_cpu_util.sh"
NET_SNAPSHOT_SCRIPT="./lib/save_net_metrics.sh"
IO_SNAPSHOT_SCRIPT="./lib/save_io_metrics.sh"

PYTHON_SCRIPT="./lib/prepare_data.py"

REPORT_SCRIPT="./lib/new_report.py"
TEMPLATE_FILE="./lib/new_template.html"
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
    echo "  This script runs the save_snapshot.sh script on all nodes of a given SLURM job."
    echo "  After the snapshot saver completes, it runs a Python script to process the data."
    echo
    echo "Example:"
    echo "  $0 12345 600"
    echo "  This runs the snapshot saver for job ID 12345 with a duration of 600 seconds."
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

# OLD CODE
# Launch one srun per node in the background
# NODE_COUNT=0
# for NODE in $NODELIST; do
#     srun --overlap --jobid="$JOB_ID" --nodes=1 --ntasks=1 -w "$NODE" bash "$GPU_SNAPSHOT_SCRIPT" --duration "$DURATION" &
#     NODE_COUNT=$((NODE_COUNT + 1))
# done

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
wait

echo "Finished data logging on $NODE_COUNT nodes."

# Start the Python script to prepare the data for analysis
echo "-------------"
echo "Starting Python script: $PYTHON_SCRIPT"
python3 "$PYTHON_SCRIPT" --job-id "$JOB_ID"

# Start the reporting script once the Python script is done
echo "-------------"
echo "Starting reporting script: $REPORT_SCRIPT"
CURRENT_DATE=$(date +%d-%m-%Y)
INPUT_FILE="./outputs/${JOB_ID}_${CURRENT_DATE}/data_${JOB_ID}_${CURRENT_DATE}_gpu.parquet"
INPUT_FILE2="./outputs/${JOB_ID}_${CURRENT_DATE}/data_${JOB_ID}_${CURRENT_DATE}.parquet"
OUTPUT_FILE="./outputs/${JOB_ID}_${CURRENT_DATE}/${JOB_ID}_report.html"
python3 "$REPORT_SCRIPT" --input_file "$INPUT_FILE" --input_file2 "$INPUT_FILE2"  --template_file "$TEMPLATE_FILE" --output_file "$OUTPUT_FILE"