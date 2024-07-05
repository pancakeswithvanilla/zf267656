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

# Plot the distribution of durations with a focus on durations between 0 and 10000, all events have 4 nucleotides
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

sorted_durations = sorted(durations)
print(sorted_durations[707]) #1287

# def binary_search(sorted_list, threshold):
#     left, right = 0, len(sorted_list) - 1
#     while left <= right:
#         mid = (left + right) // 2
#         if sorted_list[mid] <= threshold:
#             left = mid + 1
#         else:
#             right = mid - 1
#     return left

# index_exceeds_10000 = binary_search(sorted_durations, 10000)
# print(f"Index where sorted durations exceed 10000: {index_exceeds_10000}")
# for index in range (1323,1344): #3 outliers at 368136 458879 602070, investigate this later probably multiple events, #15798, pad everything at 15000
#     print(sorted_durations[index])

# Plot the signal data for the last event
last_event = event_list[220]  # Get the last event from the list
signal_data_subset = signal_data[(last_event[0]):(last_event[1])]

# Create a list of indices for x-axis
indices = list(range((last_event[0]), (last_event[1])))

# Plot the signal data
plt.figure(figsize=(10, 6))
plt.plot(indices, signal_data_subset, color='blue', marker='o', linestyle='-')
plt.xlabel('Index')
plt.ylabel('Signal Value')
plt.title('Signal Data for Last Event')
plt.grid(True)

# Save the plot to a file
plt.savefig('last_event_signal_plot.png')

print("Plot saved as 'last_event_signal_plot.png'")