#!/bin/bash

#SBATCH --job-name=sentiment_job         # Optional: name of the job
#SBATCH --time=20:00:00                  # Time limit (HH:MM:SS)
#SBATCH --account=cil_jobs

#SBATCH --output=logs/%x_%j.out          # Output file: job name and job ID
#SBATCH --error=logs/%x_%j.err           # Error file

# Run the Python script
python run_job_on_cluster.py

# To get the logs
# ls -lh logs/
# cat logs/sentiment_job_{job_id}.out
# cat logs/sentiment_job_{job_id}.err
