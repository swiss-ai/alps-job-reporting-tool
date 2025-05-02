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
#SNAPSHOT_SCRIPT="/capstor/store/cscs/swissai/a06/users/schlag/straggler_log/save_snapshot.sh"
#SNAPSHOT_SCRIPT="/users/ndegiorgi/store/straggler_log/save_snapshot.sh"
#PYTHON_SCRIPT="/users/ndegiorgi/store/straggler_log/prepare_data.py"

SNAPSHOT_SCRIPT="./scripts/save_snapshot.sh"
PYTHON_SCRIPT="./scripts/prepare_data.py"
# ===========================

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

echo "Launching $SNAPSHOT_SCRIPT on all nodes of job $JOB_ID..."

# Launch one srun per node in the background
for NODE in $NODELIST; do
    echo "  -> Launching on $NODE"
    srun --overlap --jobid="$JOB_ID" --nodes=1 --ntasks=1 -w "$NODE" bash "$SNAPSHOT_SCRIPT" --duration "$DURATION" &
done

# Wait for all background sruns to finish
wait

echo "Snapshot script launched on all nodes."

# Start the Python script to prepare the data for analysis
echo "Starting Python script: $PYTHON_SCRIPT"
python3 "$PYTHON_SCRIPT" --job-id "$JOB_ID"

echo "Python script execution completed."

