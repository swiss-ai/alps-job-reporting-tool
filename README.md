# Alps Job Reporting Tool

## Overview

The Alps Job Reporting Tool is designed to visualize the performance of nodes used in a SLURM job. It provides various graphs to help detect anomalies and identify potential issues in node performance.

## Features

- Collects performance data from nodes in a running SLURM job.
- Processes and organizes the data for easy analysis.
- Generates visualizations to highlight anomalies and performance trends.

## Setup

Follow these steps to set up the tool:

1. Clone the repository:
   ```bash
   git clone https://github.com/swiss-ai/alps-job-reporting-tool
   cd alps-job-reporting-tool
   ```

2. Create a Python environment using the `requirements.txt` file:
    - with [conda](https://docs.conda.io/):
        ```
        conda create --name myenv --file requirements.txt
        conda activate myenv
        ```
    - with [pip](https://pypi.org/project/pip/):
        ```
        python -m venv myenv
        source myenv/bin/activate  # On Windows, use myenv\Scripts\activate
        pip install -r requirements.txt
        ```


## Usage

### Collecting Data and Create a Report
To collect and process data for a given SLURM job, follow these steps:

1. Ensure the SLURM job is running.

2. Run the following command from the main directory:

   ```sh
   ./metrics_downloader.sh <job_id> [duration]
   ```

    - `<job_id>`: The SLURM job ID for which data will be collected.
    - `[duration]` (optional): The logging period in seconds for node data collection. The default is 300 seconds.
3. The collected data and an html report will be saved in the `outputs/<job_id>_<date>` folder.

### Comments

Both CSV and Parquet files are saved, allowing users to perform additional analysis and visualize the data using their preferred tools.

If there are any interesting metrics or visualization that you think could be useful let us know such that we can add them.
