import numpy as np
import pyabf
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import seaborn as sns
# Load the ABF file and extract data
abf_file = "/work/zf267656/peltfolder/dnadata/20210703 wtAeL 4M KCl A4 p2 120mV-9.abf"
signal = pyabf.ABF(abf_file)
signal_data = signal.data[0]
print("start")
# Define the threshold for clustering
threshold = 200

# Apply k-means clustering with 2 clusters based on threshold
X = signal_data.reshape(-1, 1)
kmeans = KMeans(n_clusters=2, random_state=0)
clusters = kmeans.fit_predict(X)

# Combine original indices with cluster labels to create indexed_clusters
indexed_clusters = list(zip(range(len(signal_data)), clusters))

# Separate data points into two clusters based on their label and threshold
cluster1_indices = [index for index, cluster in indexed_clusters if signal_data[index] < threshold]
cluster2_indices = [index for index, cluster in indexed_clusters if signal_data[index] >= threshold]

# Calculate means of each cluster
# mean_cluster1 = np.mean(signal_data[cluster1_indices])
# mean_cluster2 = np.mean(signal_data[cluster2_indices])

# print(f"Mean in Cluster 1 (below {threshold}): {mean_cluster1}")
# print(f"Mean in Cluster 2 (above or equal to {threshold}): {mean_cluster2}")

# # File names for cluster values and events
# file_name1 = "cluster1values.txt"
# file_name2 = "cluster2values.txt"
# events_file = "events.txt"
non_events_file = "nonevents.txt"
new_events_file = "newevents.txt"
signal_events_file = "signalevents.txt"
signal_nonevents_file = "signalnonevents.txt"
# # Function to check if the file already contains data
def is_file_already_written(file_name):
    return os.path.exists(file_name) and os.path.getsize(file_name) > 0

# # Write cluster 1 data to file and detect events if the file is not already written

# prev_index = None
# event_list = []
# start_index = cluster1_indices[0]
# end_index = None
# signal_sum = 0

# # with open(file_name1, "w") as file1:
# for index in cluster1_indices:
#     # file1.write(f"({index}, {signal_data[index]})\n")
#     signal_sum += signal_data[index]
#     if prev_index is not None and index - prev_index > 1:
#         signal_sum -= signal_data[index]
#         end_index = prev_index
#         duration = end_index - start_index
#         signal_avg = float(signal_sum // duration)
#         event_list.append([start_index, end_index, duration, signal_avg])
#         start_index = index
#         signal_sum = 0
#     if index == cluster1_indices[-1]:
#         duration = index - start_index
#         signal_avg = float(signal_sum // duration)
#         event_list.append([start_index, index, duration, signal_avg])
#     prev_index = index

prev_index = None
nonevent_list = []
start_index = cluster2_indices[0]  # Use cluster2_indices instead of cluster1_indices
end_index = None
signal_sum = 0

for index in cluster2_indices:  # Use cluster2_indices instead of cluster1_indices
    signal_sum += signal_data[index]
    if prev_index is not None and index - prev_index > 1:
        signal_sum -= signal_data[index]
        end_index = prev_index
        duration = end_index - start_index
        signal_avg = float(signal_sum // duration)
        nonevent_list.append([start_index, end_index, duration, signal_avg])
        start_index = index
        signal_sum = 0
    if index == cluster2_indices[-1]:  # Use cluster2_indices instead of cluster1_indices
        duration = index - start_index
        signal_avg = float(signal_sum // duration)
        nonevent_list.append([start_index, index, duration, signal_avg])
    prev_index = index

# Write cluster 2 data to file if the file is not already written
# if not is_file_already_written(file_name2):
#     with open(file_name2, "w") as file2:
#         for index in cluster2_indices:
#             file2.write(f"({index}, {signal_data[index]})\n")
nonevent_list = [nonevent for nonevent in nonevent_list if nonevent[3] >= 240]
new_event_list = []
for index in range(1,len(nonevent_list)) :
    start_index = nonevent_list[index -1][1]+1
    end_index = nonevent_list[index][0]-1
    duration = end_index - start_index
    new_event_list.append([start_index, end_index, duration])

new_event_list = [newevent for newevent in new_event_list if newevent[2]>= 100 and newevent[2]<= 11000]
total_events = []
total_nonevents = []
one_event =[]
one_nonevent =[]
for event in new_event_list:
    one_event =[]
    start_index = event[0]
    while start_index <= event[1]:
        one_event.append(signal_data[start_index])
        start_index = start_index + 1
    total_events.append(one_event)

for nonevent in nonevent_list:
    one_nonevent =[]
    start_index = nonevent[0]
    while start_index <= nonevent[1]:
        one_nonevent.append(signal_data[start_index])
        start_index = start_index + 1
    total_nonevents.append(one_nonevent)

if not is_file_already_written(signal_events_file):
    with open(signal_events_file, "w") as file3:
        for new_event in total_events:
            file3.write(" ".join(map(str, new_event)) + "\n")
if not is_file_already_written(signal_nonevents_file):
    with open(signal_nonevents_file, "w") as file3:
        for non_event in total_nonevents:
            file3.write(" ".join(map(str, non_event)) + "\n")

if not is_file_already_written(new_events_file):
    with open(new_events_file, "w") as file3:
        for new_event in new_event_list:
            file3.write(f"{new_event}\n")

if not is_file_already_written(non_events_file):
    with open(non_events_file, "w") as file3:
        for non_event in nonevent_list:
            file3.write(f"{non_event}\n")

            # Plot the signal data for the last event
# last_event = nonevent_list[52]  # Get the last event from the list
# signal_data_subset = signal_data[(last_event[0]):(last_event[1])]

# # Create a list of indices for x-axis
# indices = list(range((last_event[0]), (last_event[1])))

# # Plot the signal data
# plt.figure(figsize=(10, 6))
# plt.plot(indices, signal_data_subset, color='blue', marker='o', linestyle='-')
# plt.xlabel('Index')
# plt.ylabel('Signal Value')
# plt.title('Signal Data for Last Non Event')
# plt.grid(True)

# # Save the plot to a file
# plt.savefig('last_nonevent_signal_plot.png')


# event_list = [event for event in event_list if not ( (event[2] < 200 and event[3] > 150) or event[2]< 100 or event[2]>15000)]
# # Write event list to file if the file is not already written
# if not is_file_already_written(events_file):
#     with open(events_file, "w") as file3:
#         for event in event_list:
#             file3.write(f"{event}\n")

# Print event list for reference
# print(f"Total events: {len(event_list)}")
# avg_duration = 0
# max_duration = 0
# max_index = 0
# durations = []
# for index in range (len(event_list)):
#     durations.append(event_list[index][2])
#     if max_duration < event_list[index][2]:
#         max_duration = event_list[index][2]
#         max_index = index
#     avg_duration += event_list[index][2]
# avg_duration = avg_duration // len(event_list)
# print("Average duration:",avg_duration) 
# print("Max duration", max_duration,"Max_index:", max_index)

###TODO
#pad and mask duration for creating fixed length but ignoring zero values 
#create a cGAN for this time series sequence
#latent space manipulation in GAN
#DTW read and implement for feature extraction
#consider other types of architectures for task
