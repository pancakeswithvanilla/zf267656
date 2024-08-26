#!/bin/bash

# Path to your Python script
SCRIPT="/work/zf267656/peltfolder/wgan.py"

# Total number of epochs to train
total_epochs=2500

# File to keep track of the current epoch
EPOCH_FILE="/work/zf267656/peltfolder/current_epoch.txt"

# Function to check the last completed epoch
get_last_epoch() {
    if [ -f "$EPOCH_FILE" ]; then
        cat "$EPOCH_FILE"
    else
        echo 0
    fi
}

# Function to save the current epoch to a file
save_epoch() {
    echo $1 > "$EPOCH_FILE"
}

while true
do
    last_epoch=$(get_last_epoch)
    
    # Run the training script, starting from the last epoch
    python $SCRIPT --last_epoch $last_epoch --total_epochs $total_epochs

    # Check if the script finished all epochs
    if [ $? -eq 0 ]; then  # Added the missing space here
        echo "Training completed successfully. $last_epoch $total_epochs"
        break
    else
        echo "Training interrupted. Restarting from epoch $last_epoch."
        sleep 5  # Optional: Add a delay before restarting
    fi
done
