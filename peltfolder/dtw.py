from cluster import load_data, fragment_signals
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import pyabf
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.io import loadmat
import scipy
from Pelt.detection import get_events
import math
import json
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.cluster import AgglomerativeClustering
from bayes_opt import BayesianOptimization
from functools import partial
from sklearn.decomposition import PCA
import shutil

def read_last_line(output_file):
    last_i, last_j = 0, 0  
    with open(output_file, 'r') as f:
        lines = f.readlines() 
        if len(lines) >= 2: 
            last_line = lines[-1].strip().split(',') 
            last_i, last_j = int(last_line[0]), int(last_line[1])
    return last_i, last_j

def compute_dtw_distances(segments):
    output_file = 'dtw_matrix1.txt'
    num_segments = len(segments)
    dtw_matrix = np.zeros((num_segments, num_segments))
    lines_written = 0  # Track the number of lines written
    last_i, last_j = read_last_line('dtw_matrix1.txt')

    # Open the file in append mode to continue writing
    with open(output_file, 'a') as f:
        for i in range(last_i, num_segments):
            print("Index is:", i)
            for j in range((last_j + 1) if i == last_i else i + 1, num_segments):
                segment_i = np.ravel(segments[i])
                segment_j = np.ravel(segments[j])
                distance, _ = fastdtw(segment_i, segment_j, dist=lambda x, y: euclidean([x], [y]))
                
                # Store the calculated distance in the matrix
                dtw_matrix[i, j] = distance
                dtw_matrix[j, i] = distance

                # Write the current distance to the file
                f.write(f"{i},{j},{distance:.4f}\n")

                # Increment the line counter
                lines_written += 1

                # Flush the buffer every 100 lines
                if lines_written >= 1000:
                    f.flush()  # Force buffer to write to disk
                    lines_written = 0  # Reset the counter

    return dtw_matrix

def compute_within_cluster_distance(segments, labels, n_clusters, dtw_matrix):
    within_cluster_distance = 0
    for cluster in range(n_clusters):
        cluster_indices = [i for i in range(len(labels)) if labels[i] == cluster]
        if len(cluster_indices) > 1:
            cluster_distances = [dtw_matrix[i, j] for i in cluster_indices for j in cluster_indices if i != j]
            within_cluster_distance += np.mean(cluster_distances) if cluster_distances else 0
    return within_cluster_distance


def compute_between_cluster_distance(segments, labels, n_clusters):
    cluster_centroids = []
    for cluster in range(n_clusters):
        cluster_segments = [segments[i] for i in range(len(labels)) if labels[i] == cluster]
        if cluster_segments:
            cluster_centroids.append(np.mean(cluster_segments, axis=0))
    between_cluster_distance = 0
    if len(cluster_centroids) > 1:
        for i in range(len(cluster_centroids)):
            for j in range(i + 1, len(cluster_centroids)):
                distance, _ = fastdtw(cluster_centroids[i], cluster_centroids[j], dist=euclidean)
                between_cluster_distance += distance
        between_cluster_distance /= len(cluster_centroids) * (len(cluster_centroids) - 1) / 2
    return between_cluster_distance

def objective_function(n_clusters, dtw_matrix, my_events_fragments):
    n_clusters = int(n_clusters)
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
    labels = clustering_model.fit_predict(dtw_matrix)
    within_cluster_dist = compute_within_cluster_distance(my_events_fragments, labels, n_clusters, dtw_matrix)
    between_cluster_dist = compute_between_cluster_distance(my_events_fragments, labels, n_clusters)
    result = -(between_cluster_dist - within_cluster_dist)
    print(f"For {n_clusters} the total within/between cluster distance is: ", (-result))
    return result

def read_dtw_matrix(segments):
    output_file = 'dtw_matrix1.txt'
    num_segments = len(segments)
    dtw_matrix = np.zeros((num_segments, num_segments))
    computed_pairs = set()
    
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line_number, line in enumerate(f):
                i, j, dist = line.strip().split(',')
                i, j = int(i), int(j)
                computed_pairs.add((i, j))
                dtw_matrix[i, j] = float(dist)
                dtw_matrix[j, i] = float(dist)
    return dtw_matrix

def optimize_n_clusters(dtw_matrix, my_events_fragments):
    objective_partial = partial(objective_function, dtw_matrix=dtw_matrix, my_events_fragments=my_events_fragments)
    optimizer = BayesianOptimization(
        f=objective_partial,  
        pbounds={'n_clusters': (10, 100)}, 
        random_state=42,  
    )
    
    optimizer.maximize(
        init_points=10,  
        n_iter=20, 
    )
    
    best_n_clusters = optimizer.max['params']['n_clusters']
    print(f"Best n_clusters found by Bayesian Optimization: {best_n_clusters}")
    
    return best_n_clusters

def plot_representative_patterns(segments, labels, n_clusters):
    plt.figure(figsize=(10, 6))
    for cluster in range(n_clusters):
        cluster_segments = [segments[i] for i in range(len(labels)) if labels[i] == cluster]
        if cluster_segments:
            mean_pattern = np.mean(cluster_segments, axis=0)
            plt.plot(mean_pattern, label=f'Cluster {cluster + 1}')
    plt.title('Representative Patterns of Clusters')
    plt.xlabel('Time')
    plt.ylabel('Signal Value')
    plt.legend()
    plt.savefig(f"plots/clusters/cluster_of_signal_patterns_for_{n_clusters}.png")

def find_patterns_in_events():
    my_events, _, _ = load_data()
    my_events_fragments = fragment_signals(my_events, 30)
    dtw_matrix = compute_dtw_distances(my_events_fragments[0:10000])
    best_n_clusters = optimize_n_clusters(dtw_matrix, my_events_fragments)
    clustering_model = AgglomerativeClustering(n_clusters=best_n_clusters, affinity='precomputed', linkage='average')
    labels = clustering_model.fit_predict(dtw_matrix)
    plot_representative_patterns(my_events_fragments, labels, best_n_clusters)

def copy_file(source_file, destination_file):
    source_file_i, source_file_j = read_last_line(source_file)
    dest_file_i, dest_file_j = read_last_line(destination_file)
    
    if source_file_i < dest_file_i or (source_file_i == dest_file_i and source_file_j < dest_file_j):
        print(f"No new data to copy. Destination file is up-to-date.")
        return

    try:
        with open(source_file, 'r') as src, open(destination_file, 'a') as dest:
            src.seek(0, os.SEEK_SET)
            lines = src.readlines()
            
            start_line = dest_file_i * (len(lines) // source_file_i) + dest_file_j
            lines_to_copy = lines[start_line:]

            # Write the lines to the destination file
            dest.writelines(lines_to_copy)
        
        print(f"Successfully copied new content from '{source_file}' to '{destination_file}'.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

#find_patterns_in_events()
copy_file("dtw_matrix1.txt", "copy_dtw_matrix.txt")