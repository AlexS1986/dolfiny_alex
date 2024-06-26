#!/bin/bash

# Check if there are any simulation_ directories
SIMULATION_DIRS=$(ls -d simulation_* 2>/dev/null)

if [ -z "$SIMULATION_DIRS" ]; then
    echo "No simulation_ directories found."
    exit 1
fi

# Loop through each simulation directory
for DIR in $SIMULATION_DIRS
do
    if [ -d "$DIR" ]; then
        JOB_SCRIPT_PATH="$DIR/job_script.sh"
        
        # Check if the job script exists
        if [ -f "$JOB_SCRIPT_PATH" ]; then
            # Submit the job script using sbatch
            sbatch "$JOB_SCRIPT_PATH"
            
            # Check if the sbatch command was successful
            if [ $? -eq 0 ]; then
                echo "Submitted job for $DIR successfully."
            else
                echo "Failed to submit job for $DIR."
            fi
        else
            echo "Job script $JOB_SCRIPT_PATH does not exist in $DIR."
        fi
    fi
done
