#!/bin/bash
#SBATCH -A bii_dsc_community
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -c 37

./target/rivanna/install.sh

#SLURM_JOB_ID=$(sbatch my_job.sh | awk '{print $4}')

# Wait for the job to complete
# srun --jobid=$SLURMJOB_ID echo "Waiting for job $JOB_ID to complete..."
# srun --jobid=$SLURMJOB_ID sacct --format=JobID,End --noheader

#echo "Job $JOB_ID is completed."

echo "###############################################################"
echo "# squeue"
echo "###############################################################"

squeue --format="%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R" -j $SLURM_JOB_ID

echo "###############################################################"
echo "# seff"
echo "###############################################################"

seff #SLURM_JOB_ID

echo "###############################################################"
echo "# sacct analyze waittime"
echo "###############################################################"

reserved_time=$(sacct -X  -j $SLURM_JOB_ID -o Reserved --noheader)
submit_time=$(sacct -X -j $SLURM_JOB_ID -o Submit --noheader)
start_time=$(sacct -X -j $SLURM_JOB_ID -o Start --noheader)
end_time=$(sacct -X -j $SLURM_JOB_ID -o End --noheader)
cpu_time=$(sacct -X -j $SLURM_JOB_ID -o CPUTime --noheader)
elapsed_time=$(sacct -X -j $SLURM_JOB_ID -o Elapse --noheader)

# Convert timestamps to seconds since epoch
submit_seconds=$(date -d "$submit_time" +"%s")
start_seconds=$(date -d "$start_time" +"%s")
end_seconds=$(date -d "$end_time" +"%s")

# Calculate wait time in seconds
wait_time=$((start_seconds - submit_seconds))

# Convert wait time to HH:MM:SS format
wait_formatted=$(date -u -d @"$wait_time" +"%T")

echo "JobID: $SLURM_JOB_ID"
echo "CPU Time: $cpu_time"
echo "Elapsed Time: $elapsed_time"
echo "Researved Time: $reserved_time"
echo "Submit Time: $submit_time"
echo "Start Time: $start_time"
echo "End Time: $end_time"
echo "Wait Time: $wait_formatted"
echo "Wait Time in seconnds: $wait_time"

echo "###############################################################"

echo "completed"
