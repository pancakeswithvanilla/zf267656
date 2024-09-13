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
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import ListedColormap

abf_file = "/work/zf267656/peltfolder/dnadata/20210703 wtAeL 4M KCl A4 p2 120mV-9.abf"
signal = pyabf.ABF(abf_file)
signal_data = signal.data[0]
log_file_path = "/work/zf267656/peltfolder/text_outputs.txt"

def find_clusters():    
    abf_file = "/work/zf267656/peltfolder/dnadata/20210703 wtAeL 4M KCl A4 p2 120mV-9.abf"
    signal = pyabf.ABF(abf_file)
    data = signal.data[0]
    data = data.reshape(-1, 1)
    print("Length of data", len(data))
    n_clusters = 2

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(data)

    # Get cluster labels
    labels = kmeans.labels_

    # Get the cluster centroids
    centroids = kmeans.cluster_centers_
    print(centroids)

    # Calculate the distance of each point to the nearest cluster center
    _, distances = pairwise_distances_argmin_min(data, centroids)

    # Get points in each cluster
    cluster_1 = data[labels == 0]
    cluster_2 = data[labels == 1]
    #plot_cluster_histogram(cluster_1, cluster_2)
    cluster1_indices = np.where(kmeans.labels_ == 0)[0]
    cluster2_indices = np.where(kmeans.labels_ == 1)[0]
    second_cluster_distances = distances[labels == 1]
    first_cluster_distances = distances[labels == 0]
    furthest_distance = np.max(second_cluster_distances)
    mean_second_cluster_dist = np.mean(second_cluster_distances)
    mean_second_cluster = np.mean(cluster_2)
    std_dev_first_cluster = np.std(first_cluster_distances)
    std_dev_second_cluster = np.std(second_cluster_distances)
    
    lower_bound = mean_second_cluster_dist - std_dev_second_cluster
    upper_bound = mean_second_cluster_dist + std_dev_second_cluster

    # Calculate the percentage of samples within this range
    within_one_stddev = np.sum((second_cluster_distances >= lower_bound) & (second_cluster_distances <= upper_bound))
    total_samples = len(second_cluster_distances)
    percentage_within_one_stddev = (within_one_stddev / total_samples) * 100

    threshold = mean_second_cluster - std_dev_second_cluster * 2
    filtered_cluster2_indices = [index for index, value in enumerate(data) if value > threshold]
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Avg cluster_1 value: {np.mean(cluster_1):.4f}\n")
        log_file.write(f"Avg cluster_2 value: {np.mean(cluster_2):.4f}\n")
        log_file.write(f"Max distance from nonevent centroid: {furthest_distance:.4f}\n")
        log_file.write(f"Avg distance from nonevent centroid: {mean_second_cluster_dist:.4f}\n")
        log_file.write(f"Standard deviation of distances in the first cluster: {std_dev_first_cluster:.4f}\n")
        log_file.write(f"Standard deviation of distances in the second cluster: {std_dev_second_cluster:.4f}\n")
        log_file.write(f"Percentage of samples within one standard deviation: {percentage_within_one_stddev:.2f}%\n")

    return cluster1_indices, filtered_cluster2_indices

def plot_cluster_histogram(cluster_1, cluster_2):
    cluster_1_flat = np.array([val[0] for val in cluster_1])
    cluster_2_flat = np.array([val[0] for val in cluster_2])
    plt.figure()
    plt.hist(cluster_1_flat, bins=50, color='blue', alpha=0.7, label='Cluster 1')
    plt.hist(cluster_2_flat, bins=50, color='red', alpha=0.7, label='Cluster 2')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Distribution of Cluster 1 and Cluster 2')
    plt.legend()
    plt.savefig("plots/cluster/histogram_of_cluster.png")

def duration_histogram(durations, algo_name):
        # Plot the distribution of durations with a focus on durations between 0 and 10000, all events have 4 nucleotides
    plt.figure(figsize=(12, 6))

    # Histogram
    plt.hist([d for d in durations if d <= 10000], bins=50, edgecolor='k', alpha=0.7)
    plt.xlabel('Duration')
    plt.ylabel('Frequency')
    plt.title('Histogram of Event Durations')

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig(f'plots/cluster/event_duration_{algo_name}')

def is_file_already_written(file_name):
    return os.path.exists(file_name) and os.path.getsize(file_name) > 0

def find_nonevents():
    cluster1_indices, cluster2_indices = find_clusters()
    prev_index = None
    nonevent_list = []
    start_index = cluster2_indices[0] 
    end_index = None
    signal_sum = 0
    for index in cluster2_indices:  
        signal_sum += signal_data[index]
        if prev_index is not None and index - prev_index > 1:
            signal_sum -= signal_data[index]
            end_index = prev_index
            duration = end_index - start_index
            signal_avg = float(signal_sum // duration)
            if duration > 0:
                nonevent_list.append([start_index, end_index, duration, signal_avg])
            start_index = index
            signal_sum = 0
        if index == cluster2_indices[-1]:  
            duration = index - start_index
            signal_avg = float(signal_sum // duration)
            if duration > 0:
                nonevent_list.append([start_index, index, duration, signal_avg])
        prev_index = index
    return nonevent_list


def signal_avg_across_events(new_event_list):
    signal_avgs = [entry[3] for entry in new_event_list]


    mean_signal_avg = np.mean(signal_avgs)

    std_signal_avg = np.std(signal_avgs)

    lower_bound = mean_signal_avg - 4 * std_signal_avg
    upper_bound = mean_signal_avg + 4 * std_signal_avg

    outliers = [entry[3] for entry in new_event_list if entry[3] > upper_bound]
    outlier_count = len(outliers)
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Mean of signal_avg: {mean_signal_avg:.4f}\n")
        log_file.write(f"Standard deviation of signal_avg: {std_signal_avg:.4f}\n")
    

def find_events():
    new_event_list = []
    duration_list = []
    dna_data_folder = "dnadata" 
    nonevent_list = find_nonevents()
    for index in range(1,len(nonevent_list)) :
        signal_counter = 0
        start_index = nonevent_list[index -1][1]+1
        end_index = nonevent_list[index][0]-1
        duration = end_index - start_index
        signal_sum = 0
        for index in range(start_index, end_index+1):
            signal_sum += signal_data[index]
            if signal_data[index] <= 230:
                signal_counter +=1
        if duration >= 100:
            duration_list.append(duration)
        signal_avg = float(signal_sum // duration)
        if signal_counter >= 200:
            new_event_list.append([start_index, end_index, duration, signal_avg])
            file_name = f"my_events_signals.txt"
            file_path = os.path.join(dna_data_folder, file_name)
            with open(file_path, "a") as f:
                f.write(" ".join([str(signal_data[i]) for i in range(start_index, end_index + 1)]) + "\n")
    #signal_avg_across_events(new_event_list)
    #duration_histogram(duration_list, "mine")
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Events found by my algo with more than 200 signals: {len(new_event_list)}\n")
    return new_event_list

def read_signal_file(file_path):
    list_of_lists = []
    with open(file_path, "r") as f:
        for line in f:
            row_data = [float(value) for value in line.strip().split()]
            list_of_lists.append(row_data)
    return list_of_lists

def run_pelt(): #got algorithm for comparison
    samplerate = 40_000_000               #samplerate of signal
    rough_detec_params = {
        "s": 4,                         #threshhold of detecting start of event
        "e": 0,                         #threshhold of detection end of event (after start was detected)
        # "dt_exact": 50,               #number of points used to calculate local std/mean (only use when using rd_algo : exact, but generally not neccessary)
        # "rd_algo": "exact",           #if left out, use a recursive lowpass filter for thresh hold, in general leave this out
        # "lag": 1,                     #only necessary when rd_algo : exact is used. Leave it at 1 or 2, not that important
        # "max_event_length": 10e-5,    #maximum duration for a signal to be considered an event and not pore clogging for example (in SI units)
    }

    fit_method = "c_pelt"               #method to perform level fitting. Can be c_pelt, c_dynamic, cusum, pelt, dynamic
                                        #(difference between c_pelt and pelt are, that with c_pelt you can only use "model": "l2", same with c_dynamic)
                                        #c_pelt is for detecting unknown amounts of levels. Generally leave it at c_pelt or pelt.

    fit_params = {
        # "delta": 0.2,                 #parameter for cusum
        # "hbook": 1,                   #parameter for cusum
        # "sigma": 0.0387,              #parameter for cusum
        # "dt_baseline": 50,            #controlls how much of the baseline is given to perform level fitting (need some amout of baseline to see where event starts)
        "fit_event_thresh": 7,         #minimum length of entire event (in number of samples)
        "fit_level_thresh": 5,          #minimum length of one level (in number of samples)
        "pen":  100,                  #sensitivity parameter for pelt/c_pelt. Either pass float (the lower the more levels are detected) or leave it as
                                        #"BIC" or "AIC" for unsupervised level fitting. In paper more information on this.
        "model": "l2",                  #controls what type of changepoints are detected. In Documentation there is more information on this
        # "nr_ct": 5,                   #number of set changepoints if you use fit_method: dynamic/c_dynamic
    }


    show = False                        #if set to true, you can see how the algorithm detects the levels

    save_folder = "/work/zf267656/peltfolder/dnadata/"                                     #path to the folder where you want to save output
    file_name = 'c_pelt20210703 wtAeL 4M KCl A4 p2 120mV-9.abf'           #name of the output file

    num_of_events = get_events(signal_data, samplerate, rough_detec_params=rough_detec_params, fit_method=fit_method,
            fit_params=fit_params, show=show, folder_path=save_folder, filename=file_name)
    
def run_cusum(): #got algorithm for comparison
    samplerate = 40_000_000               #samplerate of signal
    rough_detec_params = {
        "s": 4,                         #threshhold of detecting start of event
        "e": 0,                         #threshhold of detection end of event (after start was detected)
        # "dt_exact": 50,               #number of points used to calculate local std/mean (only use when using rd_algo : exact, but generally not neccessary)
        # "rd_algo": "exact",           #if left out, use a recursive lowpass filter for thresh hold, in general leave this out
        # "lag": 1,                     #only necessary when rd_algo : exact is used. Leave it at 1 or 2, not that important
        # "max_event_length": 10e-5,    #maximum duration for a signal to be considered an event and not pore clogging for example (in SI units)
    }

    fit_method = "cusum"               #method to perform level fitting. Can be c_pelt, c_dynamic, cusum, pelt, dynamic
                                        #(difference between c_pelt and pelt are, that with c_pelt you can only use "model": "l2", same with c_dynamic)
                                        #c_pelt is for detecting unknown amounts of levels. Generally leave it at c_pelt or pelt.

    fit_params = {
        "delta": 0.2,                 #parameter for cusum
        "hbook": 1,                   #parameter for cusum
        "sigma": 0.0387,              #parameter for cusum
        "dt_baseline": 50,            #controlls how much of the baseline is given to perform level fitting (need some amout of baseline to see where event starts)
        "fit_event_thresh": 7,         #minimum length of entire event (in number of samples)
        "fit_level_thresh": 5,          #minimum length of one level (in number of samples)
        "pen":  100,                  #sensitivity parameter for pelt/c_pelt. Either pass float (the lower the more levels are detected) or leave it as
                                        #"BIC" or "AIC" for unsupervised level fitting. In paper more information on this.
        "model": "l2",                  #controls what type of changepoints are detected. In Documentation there is more information on this
        # "nr_ct": 5,                   #number of set changepoints if you use fit_method: dynamic/c_dynamic
    }


    show = False                        #if set to true, you can see how the algorithm detects the levels

    save_folder = "/work/zf267656/peltfolder/dnadata/"                                     #path to the folder where you want to save output
    file_name = 'cusum20210703 wtAeL 4M KCl A4 p2 120mV-9.abf'           #name of the output file

    num_of_events = get_events(signal_data, samplerate, rough_detec_params=rough_detec_params, fit_method=fit_method,
            fit_params=fit_params, show=show, folder_path=save_folder, filename=file_name)
    

def calculate_len_sig(data, algo_name):
    events = data["events"]
    length = 0
    min_length = math.inf
    max_length = 0
    counter = 0
    duration_list = []
    filtered_list_events =[]
    for index in range(len(events)):
        sig_counter = 0
        events_len = events[index]["start_end_in_raw"][1] - events[index]["start_end_in_raw"][0]
        start_signal = events[index]["start_end_in_raw"][0]
        end_signal = events[index]["start_end_in_raw"][1]
        signal_segment = signal_data[start_signal:end_signal+1]
        if 'signal_w_baseline' in events[index]:
            signal_w_baseline = events[index]['signal_w_baseline']
            if np.isnan(signal_w_baseline).any():
                continue
        for sig in signal_segment:
            if sig <= 230:
                sig_counter +=1
        if sig_counter >= 100:
            duration_list.append(sig_counter)
        if sig_counter >= 200:
            counter += 1
            filtered_list_events.append(events[index])
        if events_len < min_length:
            min_length = events_len
        if events_len > max_length:
            max_length = events_len
        length += events_len
    avg_length = length/len(events)
    #duration_histogram(duration_list, algo_name)
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Max signal length: {max_length}\n")
        log_file.write(f"Min signal length: {min_length}\n")
        log_file.write(f"Avg signal length: {avg_length}\n")
        log_file.write(f"Number of detected events: {len(events)}\n")
        log_file.write(f"Events with more than 200 signals: {counter}\n")
    return filtered_list_events


def load_data():
    pelt_data = json.load(open('/work/zf267656/peltfolder/dnadata/c_pelt20210703 wtAeL 4M KCl A4 p2 120mV-9.abf.json'))
    cusum_data = json.load(open('/work/zf267656/peltfolder/dnadata/cusum20210703 wtAeL 4M KCl A4 p2 120mV-9.abf.json'))
    file_path = "dnadata/my_events_signals.txt"
    found_by_mine = read_signal_file(file_path)
    found_by_pelt = calculate_len_sig(pelt_data, "pelt")
    found_by_cusum = calculate_len_sig(cusum_data, "cusum")
    return found_by_mine, found_by_cusum, found_by_pelt

def compare_pelt_cusum_mine():
    found_by_mine, found_by_cusum, found_by_pelt = load_data()
    comparison(found_by_mine, found_by_pelt, "pelt")
    comparison(found_by_mine, found_by_cusum, "cusum")
    

def compute_fft():
    found_by_mine,found_by_cusum ,found_by_pelt = load_data()
    nonevents_list = find_nonevents()
    apply_fft_on_samples(found_by_pelt[100:110], "pelt")
    apply_fft_on_samples(found_by_mine[30:40], "mine")
    apply_fft_on_samples(found_by_cusum[200:210], "cusum")
    apply_fft_on_samples(nonevents_list[150:175], "non_event")

def comparison(found_by_mine, found_by_algo, name_of_algo):
    counter_mine = 0
    comp_dict = {}
    counter_algo = 0
    comparison_id = 0
    overlap_index = 0
    total_events = 0
    list_of_my_indices = []
    list_of_algo_indices = []
    while counter_algo < len(found_by_algo) and counter_mine < len(found_by_mine) :
        start_algo = found_by_algo[counter_algo]["start_end_in_raw"][0]
        end_algo = found_by_algo[counter_algo]["start_end_in_raw"][1]
        start_mine = found_by_mine[counter_mine][0]
        end_mine = found_by_mine[counter_mine][1]
        if start_algo - 100 <= start_mine <= start_algo + 100:
            comp_dict[comparison_id] = {"algo": [start_algo, end_algo],"mine": [start_mine, end_mine]}
            compare_plots(start_algo, end_algo, start_mine, end_mine, comparison_id, name_of_algo)
            list_of_my_indices.append(counter_mine)
            list_of_algo_indices.append(counter_algo)
            comparison_id += 1
            counter_algo +=1
            counter_mine += 1
            total_events += 1
        else:
            if check_overlap(start_algo, end_algo, start_mine, end_mine):
                compare_plots(start_algo, end_algo, start_mine, end_mine, f"overlap_{overlap_index}", name_of_algo)
                list_of_my_indices.append(counter_mine)
                list_of_algo_indices.append(counter_algo)
                counter_algo +=1
                counter_mine += 1
                total_events += 1
            elif start_algo < start_mine:
                counter_algo +=1 
            else:
                counter_mine +=1
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Total found events:{total_events}")
    plot_not_found_events(list_of_my_indices, list_of_algo_indices, found_by_algo, found_by_mine, name_of_algo)

def plot_not_found_events(list_of_my_indices, list_of_algo_indices, found_by_algo, found_by_mine, name_of_algo):
    missing_numbers_algo, missing_numbers_mine = find_single_events(list_of_my_indices, list_of_algo_indices, found_by_mine, found_by_algo, name_of_algo)
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Not found  my events in total: {len(missing_numbers_mine)}")
        log_file.write(f"Not found  {name_of_algo} events in total: {len(missing_numbers_algo)}")
    for index in range (len(missing_numbers_algo)):
        counter_algo = missing_numbers_algo[index]
        start_algo = found_by_algo[counter_algo]["start_end_in_raw"][0]
        end_algo = found_by_algo[counter_algo]["start_end_in_raw"][1]
        plot_single_events(start_algo, end_algo, f"{name_of_algo}_{counter_algo}", name_of_algo)
    for index in range (len(missing_numbers_mine)):
        counter_mine = missing_numbers_mine[index]
        start_mine = found_by_mine[counter_mine][0]
        end_mine = found_by_mine[counter_mine][1]
        plot_single_events(start_mine, end_mine, f"mine_{counter_mine}", name_of_algo)
    
def find_single_events(list_of_my_indices, list_of_algo_indices, found_by_mine, found_by_algo, name_of_algo):
    len_algo = len(found_by_algo)
    len_mine = len(found_by_mine)
    expected_range_algo = set(range(len_algo))
    numbers_set_algo = set(list_of_algo_indices)
    missing_numbers_algo = expected_range_algo - numbers_set_algo
    missing_numbers_algo = sorted(missing_numbers_algo)
    expected_range_mine = set(range(len_mine))
    numbers_set_mine = set(list_of_my_indices)
    missing_numbers_mine = expected_range_mine - numbers_set_mine
    missing_numbers_mine = sorted(missing_numbers_mine)
    # print(f"Missing numbers {name_of_algo}", missing_numbers_algo)
    # print("Missing numbers mine", missing_numbers_mine)
    return missing_numbers_algo, missing_numbers_mine


def check_overlap(start_algo, end_algo, start_mine, end_mine):
    if start_mine <= end_algo and end_mine >= start_algo:
        return True
    if start_algo <= end_mine and end_algo >= start_mine:
        return True   
    return False  

def compare_plots(start_algo, end_algo, start_mine, end_mine, comp_id, name_of_algo):
    
    # Extract the data for the specific ranges
    pelt_data = signal_data[start_algo:end_algo + 1]  # +1 to include end index
    mine_data = signal_data[start_mine:end_mine + 1]  # +1 to include end index
    
    # Create a plot
    plt.figure()
    
    # Plot the data for pelt in red
    plt.plot(range(start_algo, end_algo + 1), pelt_data, color='red', label=f'{name_of_algo} Data')
    
    # Plot the data for mine in blue
    plt.plot(range(start_mine, end_mine + 1), mine_data, color='blue', label='Mine Data')
    
    # Add titles and labels
    plt.title(f"Comparison for Comp ID: {comp_id}")
    plt.xlabel("Index")
    plt.ylabel("Data Value")
    
    # Add legend
    plt.legend()
    # Save the plot to the plots folder
    plt.savefig(f"plots/{name_of_algo}/comparison_plot_{comp_id}.png")


def plot_single_events(start_algo, end_algo, comp_id, name_of_algo):
    
    # Extract the data for the specific ranges
    algo_data = signal_data[start_algo:end_algo + 1]  # +1 to include end index
    
    # Create a plot
    plt.figure()
    
    # Plot the data for pelt in red
    plt.plot(range(start_algo, end_algo + 1), algo_data, color='red', label=f'{name_of_algo} Data')
    
    # Add titles and labels
    plt.title(f"Single Plot for {comp_id}")
    plt.xlabel("Index")
    plt.ylabel("Data Value")
    
    # Add legend
    plt.legend()
    # Save the plot to the plots folder
    plt.savefig(f"events_with_no_comp/{name_of_algo}/single_plot_{comp_id}.png")

def apply_fft_on_samples(sample_lists, algo_name):
    fft_results = []
    for idx, samples in enumerate(sample_lists):
        if algo_name == "mine" or algo_name =="non_event":
            start = samples[0]
            end = samples[1]
        else:
            start = samples["start_end_in_raw"][0]
            end = samples["start_end_in_raw"][1]
        signal = signal_data[start:end]
        #print("signal start", signal_data[start], "signal_end", signal_data[end])
        
        fft_result = np.fft.fft(signal)
        fft_results.append(fft_result)
        
        plt.figure()
        fft_magnitude = np.abs(fft_result) 
        plt.plot(fft_magnitude)
        plt.title(f"FFT Result for Sample {idx + 1}")
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        
        plt.ylim([0, 10000])  
        plt.savefig(f"plots/fft_samples/{algo_name}/fft_plot_{idx + 1}.png")
        plt.close()  

#gan losses computen
        
def fragment_signals(signals, desired_length):
    """
    Fragment signals into smaller segments of a given desired length.

    Args:
    - signals: List of signals or dictionaries containing 'signal_w_baseline' or just raw signal values.
    - desired_length: Desired length of each fragment.

    Returns:
    - fragmented_signals: Numpy array of fragmented signals.
    """
    fragmented_signals = []
    
    for signal in signals:
        # Check if the signal is a dictionary or a list of raw values
        if isinstance(signal, dict) and 'signal_w_baseline' in signal:
            signal_data = signal['signal_w_baseline']
        elif isinstance(signal, list) or isinstance(signal, np.ndarray):
            signal_data = signal
        else:
            print(f"Invalid signal format: {signal}")
            continue
        
        # Fragment the signal
        num_fragments = math.ceil(len(signal_data) / desired_length)
        for i in range(num_fragments):
            start_idx = i * desired_length
            end_idx = start_idx + desired_length
            
            # Adjust end_idx to prevent going out of bounds
            if end_idx > len(signal_data):
                end_idx = len(signal_data)
                start_idx = len(signal_data) - desired_length  # Adjust start for last fragment
            
            # Extract the fragment
            fragment = signal_data[start_idx:end_idx]
            fragmented_signals.append(fragment)
    
    return np.array(fragmented_signals)


def pca_for_events():
    my_events, cusum_events, pelt_events = load_data()
    all_events = {"my_events":my_events, "cusum_events":cusum_events, "pelt_events":pelt_events}
    segments_list_len = [50, 100, 200]
    n_comp_list = [5]
    for key  in all_events:
        event = all_events.get(key)
        min = math.inf
        for event in my_events:
            if min > len(event):
                min = len(event)
        print(f"For {key} min length of an event is: ", min)
    for segment_len in segments_list_len:
        for key in all_events:
            events = all_events.get(key)
            my_events_fragments = fragment_signals(events, segment_len)
            for n_components in n_comp_list:
                pca = PCA(n_components=n_components)
                pca.fit(my_events_fragments)
                explained_variance = pca.explained_variance_
                explained_variance_sum = np.sum(explained_variance)

                plt.figure(figsize=(8, 6))
                plt.bar(range(1, (n_components+1)), explained_variance)
                plt.axhline(y=explained_variance_sum, color='r', linestyle='--', label=f'Sum of Explained Variance: {explained_variance_sum:.2f}')
                plt.xlabel('Principal Components')
                plt.ylabel('Explained Variance')
                plt.title(f'Explained Variance by PCA Components for {n_components} features for {key} algo data for data of length {segment_len}')
                plt.legend()
                plt.savefig(f"plots/pca/expl_var_ration_{key}_{n_components}_{segment_len}.png")

def preprocess_data(data):
    """
    Preprocess the data by scaling.
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

def apply_tsne(data, n_components=2, random_state=0):
    """
    Apply t-SNE to the data and reduce dimensions.
    """
    tsne = TSNE(n_components=n_components, random_state=random_state)
    tsne_results = tsne.fit_transform(data)
    return tsne_results

def plot_tsne(tsne_results, labels=None):
    """
    Plot the t-SNE results with specific colors for labels.
    """
    plt.figure(figsize=(10, 8))
    
    # Define the colormap
    custom_cmap = ListedColormap(['red', 'blue'])  # Red for label 0, Blue for label 1

    if labels is not None:
        # Ensure that the labels array contains only 0s and 1s
        if not np.all(np.isin(labels, [0, 1])):
            raise ValueError("Labels must contain only 0s and 1s.")
        
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap=custom_cmap, alpha=0.7)
        plt.colorbar(scatter, label='Event Category', ticks=[0, 1], format=plt.FuncFormatter(lambda x, _: 'Non-Event' if x == 0 else 'Event'))
    else:
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.7)
    
    plt.title('t-SNE Visualization of Fragmented Signals')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig("plots/tsne/events_nonevents_my_algo.png")

def tsne():
    my_data, _ , _ = load_data()
    non_events = find_nonevents()
    noneventsiglist = []
    for nonevent in non_events:
        start_index = nonevent[0]
        end_index = nonevent[1]
        noneventsignal = []
        for index in range(start_index, end_index):
            noneventsignal.append(signal_data[index])
        noneventsiglist.append(noneventsignal)
    trimmed_nonevents = [noneventel for noneventel in noneventsiglist if len(noneventel) >=200]
    trimmed_nonevents = fragment_signals(trimmed_nonevents, 200)
    my_events_fragments = fragment_signals(my_data, 200)
    trimmed_nonevents_sample = trimmed_nonevents[0:len(my_events_fragments)]
    labels_non_events = [0] * len(trimmed_nonevents_sample)
    labels_events = [1] * len(my_events_fragments)
    combined_fragments = np.vstack([trimmed_nonevents_sample, my_events_fragments])
    combined_labels = labels_non_events + labels_events
    print("trimmed nonevents length", len(trimmed_nonevents_sample), "trimmed my_events", len(my_events_fragments))
    print(f"Length of combined_fragments: {len(combined_fragments)}")
    print(f"Length of combined_labels: {len(combined_labels)}")
    scaled_data = preprocess_data(combined_fragments)
    tsne_results = apply_tsne(scaled_data)
    plot_tsne(tsne_results, labels=combined_labels)

#tsne()

#copy_file()
        #timing comparison of dif algos
        #struktur bachelorthesis anfangen schreiben
        #timegan weiterentwickeln
        #benchmark multiple freiburg