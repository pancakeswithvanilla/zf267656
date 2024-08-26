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
    print("Avg cluster_2 value: ", np.mean(cluster_2))
    print("Avg cluster_1 value: ", np.mean(cluster_1))
    second_cluster_distances = distances[labels == 1]
    first_cluster_distances = distances[labels == 0]
    furthest_distance = np.max(second_cluster_distances)
    print("Max distance from nonevent centroid: ", furthest_distance)
    mean_second_cluster = np.mean(second_cluster_distances)
    print("Avg distance from nonevent centroid: ",mean_second_cluster)
    std_dev_first_cluster = np.std(first_cluster_distances)
    std_dev_second_cluster = np.std(second_cluster_distances)
    print(f"Standard deviation of distances in the first cluster: {std_dev_first_cluster:.4f}")
    print(f"Standard deviation of distances in the second cluster: {std_dev_second_cluster:.4f}")
    
    lower_bound = mean_second_cluster - std_dev_second_cluster
    upper_bound = mean_second_cluster + std_dev_second_cluster

    # Calculate the percentage of samples within this range
    within_one_stddev = np.sum((second_cluster_distances >= lower_bound) & (second_cluster_distances <= upper_bound))
    total_samples = len(second_cluster_distances)
    percentage_within_one_stddev = (within_one_stddev / total_samples) * 100

    print(f"Percentage of samples within one standard deviation: {percentage_within_one_stddev:.2f}%")
    return cluster1_indices, cluster2_indices


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


def find_events():
    new_event_list = []
    nonevent_list = find_nonevents()
    for index in range(1,len(nonevent_list)) :
        start_index = nonevent_list[index -1][1]+1
        end_index = nonevent_list[index][0]-1
        duration = end_index - start_index
        signal_sum = 0
        for index in range(start_index, end_index+1):
            signal_sum += signal_data[index]
        signal_avg = float(signal_sum // duration)
        new_event_list.append([start_index, end_index, duration, signal_avg])
        new_event_trimmed = [newevent for newevent in new_event_list if newevent[2]>= 200]
    print("Events found by my algo:", len(new_event_list))
    print("Events found by my algo with more than 200 signals:", len(new_event_trimmed))
    return new_event_trimmed

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

def calculate_len_sig():
    run_pelt()
    data = json.load(open('/work/zf267656/peltfolder/dnadata/c_pelt20210703 wtAeL 4M KCl A4 p2 120mV-9.abf.json'))
    events = data["events"]
    length = 0
    min_length = math.inf
    max_length = 0
    counter = 0
    filtered_list_events =[]
    for index in range(len(events)):
        events_len = len(events[index]["signal_w_baseline"])
        if events_len >= 200:
            counter += 1
            filtered_list_events.append(events[index])
        if events_len < min_length:
            min_length = events_len
        if events_len > max_length:
            max_length = events_len
        length += events_len
    avg_length = length/len(events)
    print("Max signal length: " ,max_length)
    print("Min signal length: " ,min_length)
    print("Avg signal length: " ,avg_length)
    print("Number of detected events: ", len(events))
    print("Events with more than 200 signals: ", counter)
    return filtered_list_events

def comparison():
    found_by_mine = find_events()
    found_by_pelt = calculate_len_sig()
    for index in range(10):
        print("Event start of pelt, event end of pelt:", found_by_pelt[index]["start_end_in_raw"])
        print("Event length of pelt:", (found_by_pelt[index]["start_end_in_raw"][1]- found_by_pelt[index]["start_end_in_raw"][0]))
        print("Event start of mine, event end of mine:", found_by_mine[index][0],found_by_mine[index][1])
        print("Event length of mine: ", found_by_mine[index][2])

    for index in range (len(found_by_pelt)):
        start_pelt = found_by_pelt[index]["start_end_in_raw"][0]
        end_pelt = found_by_pelt[index]["start_end_in_raw"][1]
        start_mine = found_by_mine[index][0]
        end_mine = found_by_mine[index][1]
        comp_dict = {}
        if start_pelt - 100 <= start_mine <= start_pelt + 100:
            comp_dict["pelt"] = [start_pelt, end_pelt]
            comp_dict["mine"] = [start_mine, end_mine]


comparison()