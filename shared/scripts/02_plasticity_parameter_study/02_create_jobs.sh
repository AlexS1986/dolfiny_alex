#!/bin/bash

# Check if job template exists
JOB_TEMPLATE="00_jobs/job_template.sh"

if [ ! -f "$JOB_TEMPLATE" ]; then
    echo "Job template $JOB_TEMPLATE does not exist."
    exit 1
fi

# Determine the total number of simulation directories
TOTAL_COMPUTATIONS=$(ls -d simulation_* 2>/dev/null | wc -l)

if [ "$TOTAL_COMPUTATIONS" -eq 0 ]; then
    echo "No simulation_ directories found."
    exit 1
fi

echo "Found $TOTAL_COMPUTATIONS simulation_ directories."

# Loop through each simulation directory
for DIR in simulation_*
do
    if [ -d "$DIR" ]; then
        # Extract the computation number from the directory name
        CURRENT_COMPUTATION=$(echo $DIR | grep -o -E '[0-9]+')

        # Define the job script path
        JOB_SCRIPT_PATH="$DIR/job_script.sh"

        # Replace placeholders in the template and write to the job script
        sed -e "s/{FOLDER_NAME}/$DIR/g" \
            -e "s/{TOTAL_COMPUTATIONS}/$TOTAL_COMPUTATIONS/g" \
            -e "s/{CURRENT_COMPUTATION}/$CURRENT_COMPUTATION/g" \
            -e "s/{JOB_NAME}/$DIR/g" \
            "$JOB_TEMPLATE" > "$JOB_SCRIPT_PATH"

        # Check if the job script was created successfully
        if [ $? -eq 0 ]; then
            echo "Created $JOB_SCRIPT_PATH successfully."
        else
            echo "Failed to create $JOB_SCRIPT_PATH."
        fi
    fi
done


