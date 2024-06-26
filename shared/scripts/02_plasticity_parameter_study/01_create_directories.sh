#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 TOTAL_COMPUTATIONS"
    exit 1
fi

# Get the total number of computations from the argument
TOTAL_COMPUTATIONS=$1

# Loop through the range from 1 to TOTAL_COMPUTATIONS
for (( i=1; i<=TOTAL_COMPUTATIONS; i++ ))
do
    # Create the new directory name
    NEW_DIR="simulation_$i"
    
    # Copy the 000_template directory to the new directory
    cp -r 000_template "$NEW_DIR"
    
    # Check if the copy was successful
    if [ $? -eq 0 ]; then
        echo "Created $NEW_DIR successfully."
    else
        echo "Failed to create $NEW_DIR."
    fi
done
