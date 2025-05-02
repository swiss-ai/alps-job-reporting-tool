# Alps job reporting tool


## Introduction

Tool used to visualize the performance of the nodes used in a job. With different graphs that should help detect anomalies and possible problems.


## Setup

Setup by following those steps:
- clone repository
- create an environment using the `requirements.txt` file
    - with conda this would be: `conda create --name myenv --file requirements.txt`



## Usage

How to download the data for a given *job_id*
- call `submit_snapshot_saver.sh <job_id> [duration]` to start the data download and processing of a **running job**
- use the saved data from the /outputs/<job_id> folder