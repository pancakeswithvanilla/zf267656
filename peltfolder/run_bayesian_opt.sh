#!/bin/bash

# Path to your Python script
SCRIPT="/work/zf267656/peltfolder/bayesianopt.py"

# Minimum number of rows required in the file
MIN_ROWS=100

# Counter for tracking interruptions or errors
interrupt_count=0
# Function to count rows in the file
count_rows() {
    if [ -f "discriminator_accuracy.txt" ]; then
        wc -l < "discriminator_accuracy.txt"
    else
        echo 0
    fi
}

while true
do
    python $SCRIPT
    exit_code=$?

    # Check if the script ended prematurely due to any error
    if [ $exit_code -ne 0 ]; then
        interrupt_count=$((interrupt_count+1))
        echo "Script ended with exit code $exit_code. Attempt $interrupt_count."
        
        # Optional: Add a sleep period before retrying to avoid rapid restarts
        sleep 10
    else
        echo "Script completed successfully."
        
        # Check if the file meets the row requirement
        row_count=$(count_rows)
        if [ $row_count -ge $MIN_ROWS ]; then
            echo "File contains $row_count rows, which is sufficient. Stopping."
            break
        else
            echo "File contains $row_count rows, less than the required $MIN_ROWS. Continuing..."
            
            # Optional: Add a sleep period before retrying to avoid rapid restarts
            sleep 10
        fi
    fi
done

echo "Reached the desired number of rows or stopped due to maximum retries."
