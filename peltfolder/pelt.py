import numpy as np
import pyabf
from sklearn.cluster import KMeans

# Load the ABF file and extract data
abf_file = "/work/zf267656/peltfolder/dnadata/20210703 wtAeL 4M KCl A4 p2 120mV-9.abf"
signal = pyabf.ABF(abf_file)
signal_data = signal.data[0]

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
mean_cluster1 = np.mean(signal_data[cluster1_indices])
mean_cluster2 = np.mean(signal_data[cluster2_indices])

print(f"Mean in Cluster 1 (below {threshold}): {mean_cluster1}")
print(f"Mean in Cluster 2 (above or equal to {threshold}): {mean_cluster2}")

# Print elements from each cluster with their original indices to files
file_name1 = "cluster1values.txt"
file_name2 = "cluster2values.txt"
prev_index = None
event_list = []
start_index = cluster1_indices[0]
end_index = None
signal_sum = 0

# Write cluster 1 data to file and detect events
with open(file_name1, "w") as file1:
    for index in cluster1_indices:
        file1.write(f"({index}, {signal_data[index]})\n")
        signal_sum += signal_data[index]
        if prev_index is not None and index - prev_index > 1:
            signal_sum -= signal_data[index]
            end_index = prev_index
            duration = end_index - start_index
            signal_avg = float(signal_sum // duration)
            event_list.append([start_index, end_index, duration, signal_avg])
            start_index = index
            signal_sum = 0
        if index == cluster1_indices[-1]:
            duration = index - start_index
            signal_avg = float(signal_sum // duration)
            event_list.append([start_index, index, duration, signal_avg])
        prev_index = index

# Write cluster 2 data to file
with open(file_name2, "w") as file2:
    for index in cluster2_indices:
        file2.write(f"({index}, {signal_data[index]})\n")

# Calculate lowest value in cluster 2 and highest value in cluster 1
lowest_value = np.min(signal_data[cluster2_indices])
highest_value = np.max(signal_data[cluster1_indices])
print(f"Lowest value in Cluster 2 (above or equal to {threshold}): {lowest_value}")
print(f"Highest value in Cluster 1 (below {threshold}): {highest_value}")

# Print event list for reference
print("Event List:")
for event in event_list:
    print(event)

