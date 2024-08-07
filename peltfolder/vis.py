import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pyabf
import math
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
events_file = "newevents.txt"
event_list = read_event_list(events_file)
nonevents_file = "nonevents.txt"
nonevent_list = read_event_list(nonevents_file)

# Print event list for reference
print(f"Total events: {len(nonevent_list)}")
durations = []

def calculate_durations(event_list):
    avg_duration = 0
    max_duration = 0
    max_index = 0
    for index in range(len(event_list)):
        durations.append(event_list[index][2])
        if max_duration < event_list[index][2]:
            max_duration = event_list[index][2]
            max_index = index
        avg_duration += event_list[index][2]
    avg_duration = avg_duration // len(event_list)
    return max_index, max_duration, avg_duration

max_event_ind, max_event_dur, avg_event_dur = calculate_durations(event_list)
print(" Max event duration:", max_event_dur)
print(" Avg event duration:", avg_event_dur)
max_nonevent_ind, max_nonevent_dur, avg_nonevent_dur = calculate_durations(nonevent_list)
print(" Max nonevent duration:", max_nonevent_dur)
print(" Avg nonevent duration:", avg_nonevent_dur)


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
last_event = event_list[50]  # Get the last event from the list
signal_data_subset = signal_data[(last_event[0]-100):(last_event[1]+100)]

# Create a list of indices for x-axis
indices = list(range((last_event[0]-100), (last_event[1]+100)))

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

events_name = "signalevents.txt"
fragmented_events_name = "fragsigev.txt"
nonevents_name ="signalnonevents.txt"
fragmented_non_events_name = "fragsignonev.txt"
signals = []
# Function to read signals from a file
def read_signals(file_name):
    signals = []
    with open(file_name, 'r') as file:
        for line in file:
            signal = line.split()
            signals.append(signal)
    return signals

# Function to pad signals to a desired length
def pad_signals(signals, desired_length):
    padded_signals = []
    for signal in signals:
        if len(signal) < desired_length:
            # Pad the signal with zeros
            padded_signal = np.pad(signal, (0, desired_length - len(signal)), mode='constant', constant_values=0)
        else:
            padded_signal = signal
        padded_signals.append(padded_signal)
    return padded_signals

def fragment_signals(signals, desired_length):
    fragmented_signals = []
    for signal in signals:
        # Determine the number of fragments needed
        num_fragments = math.ceil(len(signal) / desired_length)
        
        # Split the signal into fragments
        for i in range(num_fragments):
            start_idx = i * desired_length
            end_idx = start_idx + desired_length 
            
            # Handle the case where the end index exceeds the signal length
            if end_idx > len(signal):
                end_idx = len(signal)
                start_idx = len(signal) - desired_length  # Ensure last fragment has the correct length
                
            
            fragment = signal[start_idx:end_idx]
            
            fragmented_signals.append(fragment)
    
    return np.array(fragmented_signals)

# Function to write padded signals back to the file
def write_signals(file_name, signals):
    """
    Write the first 'max_rows' signals to a file.
    
    :param file_name: The name of the file to write the signals to.
    :param signals: A list of signals (arrays or lists) to be written.
    :param max_rows: Maximum number of rows to write. Default is 10.
    """
    with open(file_name, 'w') as file:
        for i, signal in enumerate(signals):
            # if i >= max_rows:
            #     break
            # Convert the numpy array or list to a space-separated string and write it to the file
            file.write(' '.join(map(str, signal)) + '\n')


def check_line_length(file_name):
    total_length = 0
    with open(file_name, 'r') as file:
        for line in file:
            numbers = line.split()
            if len(numbers) == desired_length:
                    total_length = total_length + 1
    return total_length

def read_signals_from_generated(file_name):
    signals = []
    with open(file_name, 'r') as file:
        current_signal = []
        for line in file:
            stripped_line = line.strip()
            if stripped_line == "":
                if current_signal:
                    signals.append(current_signal)
                    current_signal = []
            else:
                current_signal.extend(map(float, stripped_line.split()))
        if current_signal:  # To handle the last signal if no trailing newline exists
            signals.append(current_signal)
    return np.array(signals)
def find_shortest_length(signals):
    # Find the length of the shortest list
    min_length = min(len(signal) for signal in signals)
    return min_length
def plot_signal(signal, output_file_name):
    """
    Plot the signal data and save the plot to a file.

    Args:
        signal (np.ndarray): The signal data to be plotted.
        output_file_name (str): The file name to save the plot.
    """
    signal_values = np.array(signal, dtype=float)

    # Create a new set of indices from 0 to 100
    num_points = len(signal_values)
    indices = np.linspace(0, 100, num_points)

# Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(indices, signal_values, marker='o')
    plt.xlabel('Index')
    plt.ylabel('Signal Value')
    plt.title('Signal Data')
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(output_file_name)
    plt.close()


# desired_length = 100
directory = "/work/zf267656/peltfolder/plots"
# signals = read_signals(events_name)
# fragmented_signals = fragment_signals(signals, desired_length)
# print(fragmented_signals.shape)
# write_signals(fragmented_events_name, fragmented_signals)
# # for index in range (10):
# #     output_file_name = f"generated_signalevent{index+1}.png"
# #     output_file_path = os.path.join(directory, output_file_name)
# #     plot_signal(fragmented_signals[index], output_file_path)

# signals = read_signals(nonevents_name)
# print("Shortest list:",find_shortest_length(signals))
# fragmented_signals = fragment_signals(signals, desired_length)
# print(fragmented_signals.shape)
# write_signals(fragmented_non_events_name, fragmented_signals)
# # # # fragmented_padded_signals = pad_signals(fragmented_signals, desired_length)
# # # # write_signals(nonevents_name, fragmented_padded_signals)
# # total_length= check_line_length(fragmented_non_events_name)
# # print("length of nonevents is equal:",total_length)
# # for index in range (10):
# #     output_file_name = f"generated_signalnonevent{index+1}.png"
# #     output_file_path = os.path.join(directory, output_file_name)
# #     plot_signal(fragmented_signals[index], output_file_path)
# # # signals = read_signals(events_name)
# # # padded_signals = pad_signals(signals, desired_length)
# # # write_signals(events_name, padded_signals)
# # # total_length= check_line_length(events_name)
# # # print("length of nonevents is equal:",total_length)

gen_sig = read_signals_from_generated("generated_samples_epoch_5000.txt")
print(gen_sig.shape)
output_file_name = f"gan_signal5000[0].png"
output_file_path = os.path.join(directory, output_file_name)
plot_signal(gen_sig[0], output_file_path)

