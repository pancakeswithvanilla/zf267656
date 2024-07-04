import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pyabf

# Load signal data
abf_file = "/work/zf267656/peltfolder/dnadata/20210703 wtAeL 4M KCl A4 p2 120mV-9.abf"
signal = pyabf.ABF(abf_file)
signal_data = signal.data[0]

# Function to read event list from file
def read_event_list(file_name):
    event_list = []
    with open(file_name, 'r') as file:
        for line in file:
            event = eval(line.strip())
            event_list.append(event)
    return event_list

# Read event list from file
events_file = "events.txt"
event_list = read_event_list(events_file)

# Print event list for reference
print(f"Total events: {len(event_list)}")
avg_duration = 0
max_duration = 0
max_index = 0
durations = []

for index in range(len(event_list)):
    durations.append(event_list[index][2])
    if max_duration < event_list[index][2]:
        max_duration = event_list[index][2]
        max_index = index
    avg_duration += event_list[index][2]

avg_duration = avg_duration // len(event_list)
print("Average duration:", avg_duration)
print("Max duration", max_duration, "Max index:", max_index)

# Plot the distribution of durations with a focus on durations between 0 and 10000
plt.figure(figsize=(12, 6))

# Histogram
plt.hist([d for d in durations if d <= 10000], bins=50, edgecolor='k', alpha=0.7)
plt.xlabel('Duration')
plt.ylabel('Frequency')
plt.title('Histogram of Event Durations (0 to 10000)')

# Save the plot to a file
plt.tight_layout()
plt.savefig('event_durations_distribution.png')

print("Histogram saved as 'event_durations_distribution.png'")

