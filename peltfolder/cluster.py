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
    with open(log_file_path, "w") as log_file:
        log_file.write(f"Avg cluster_1 value: {np.mean(cluster_1):.4f}\n")
        log_file.write(f"Avg cluster_2 value: {np.mean(cluster_2):.4f}\n")
        log_file.write(f"Max distance from nonevent centroid: {furthest_distance:.4f}\n")
        log_file.write(f"Avg distance from nonevent centroid: {mean_second_cluster_dist:.4f}\n")
        log_file.write(f"Standard deviation of distances in the first cluster: {std_dev_first_cluster:.4f}\n")
        log_file.write(f"Standard deviation of distances in the second cluster: {std_dev_second_cluster:.4f}\n")
        log_file.write(f"Percentage of samples within one standard deviation: {percentage_within_one_stddev:.2f}%\n")

    return cluster1_indices, filtered_cluster2_indices


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
            nonevent_list.append([start_index, end_index, duration, signal_avg])
            start_index = index
            signal_sum = 0
        if index == cluster2_indices[-1]:  
            duration = index - start_index
            signal_avg = float(signal_sum // duration)
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
        signal_avg = float(signal_sum // duration)
        if signal_counter >= 200:
            new_event_list.append([start_index, end_index, duration, signal_avg])
    #new_event_trimmed = [newevent for newevent in new_event_list if newevent[2]>= 150]
    signal_avg_across_events(new_event_list)
        #new_event_trimmed = [newevent for newevent in new_event_list if newevent[3]>= 220]
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Events found by my algo with more than 200 signals: {len(new_event_list)}\n")
    return new_event_list

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
    

def calculate_len_sig(data):
    events = data["events"]
    length = 0
    min_length = math.inf
    max_length = 0
    counter = 0
    filtered_list_events =[]
    for index in range(len(events)):
        sig_counter = 0
        events_len = events[index]["start_end_in_raw"][1] - events[index]["start_end_in_raw"][0]
        start_signal = events[index]["start_end_in_raw"][0]
        end_signal = events[index]["start_end_in_raw"][1]
        signal_segment = signal_data[start_signal:end_signal+1]
        for sig in signal_segment:
            if sig <= 230:
                sig_counter +=1
        if sig_counter >= 200:
            counter += 1
            filtered_list_events.append(events[index])
        if events_len < min_length:
            min_length = events_len
        if events_len > max_length:
            max_length = events_len
        length += events_len
    avg_length = length/len(events)
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
    found_by_mine = find_events()
    found_by_pelt = calculate_len_sig(pelt_data)
    found_by_cusum = calculate_len_sig(cusum_data)
    return found_by_mine, found_by_cusum, found_by_pelt

def compare_pelt_cusum_mine():
    found_by_mine, found_by_cusum, found_by_pelt = load_data()
    comparison(found_by_mine, found_by_pelt, "pelt")
    comparison(found_by_mine, found_by_cusum, "cusum")
    

def compute_fft_for_my_events():
    found_by_mine,_ ,_ = load_data()
    apply_fft_on_samples(found_by_mine)

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
        log_file.write(f"Total found events:{ total_events}")
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

def apply_fft_on_samples(sample_lists):
    fft_results = []
    
    # Loop through each list of samples
    for idx, samples in enumerate(sample_lists):
        if idx >= 10:  # Only plot the first 10 results
            break
        
        # Extract signal data based on start and end indices
        start_mine = samples[0]
        end_mine = samples[1]
        signal = signal_data[start_mine:end_mine]
        print("signal start", signal_data[start_mine], "signal_end", signal_data[end_mine])
        
        fft_result = np.fft.fft(signal)
        fft_results.append(fft_result)
        
 
        plt.figure()
        fft_magnitude = np.abs(fft_result) 
        plt.plot(fft_magnitude)
        plt.title(f"FFT Result for Sample {idx + 1}")
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        
        # Save the plot to the plots folder
        plt.savefig(f"plots/fft_samples/fft_plot_{idx + 1}.png")
        plt.close()  # Close the plot to free up memory




#compare_pelt_cusum_mine()
compute_fft_for_my_events()