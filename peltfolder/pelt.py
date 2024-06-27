import numpy as np
import matplotlib.pyplot as plt
import os
from Pelt.detection import get_events  

import os

folder_path = '/work/zf267656/peltfolder/dnadata'
filename = '20210703 wtAeL 4M KCl A4 p2 120mV-9.abf'
file_path = os.path.join(folder_path, filename)

if not os.path.isfile(file_path):
    raise FileNotFoundError(f"File '{file_path}' not found.")

data = np.fromfile(file_path, dtype=np.float32)

samplerate = 10000  
rough_detec_params = {
    "rd_algo": "exact",  
    "dt_exact": 100,     
    "s": 5,
    "e": 0,
    "max_event_length": 0.5,
    "lag": 2
}
fit_params = {
    "dt_baseline": 50,    
    "fit_method": "pelt",
    "fit_event_thresh": 10,
    "fit_level_thresh": 7
}
fit_method = "pelt"  

events = get_events(data, samplerate, fit_params, fit_method, rough_detec_params)

for event in events:
    signal = event['signal_w_baseline']
    start, end = event['start_end_in_sig']
    local_baseline = event['local_baseline']

    time = np.arange(len(signal)) / samplerate

    plt.figure(figsize=(10, 6))
    plt.plot(time, signal, label='Corrected Signal')
    plt.axhline(local_baseline, color='r', linestyle='--', label='Local Baseline')
    plt.axvline(time[start], color='g', linestyle='--', label='Event Start')
    plt.axvline(time[end], color='g', linestyle='--', label='Event End')
    plt.xlabel('Time (s)')
    plt.ylabel('Current (nA)')
    plt.title('Event Signal Visualization')
    plt.legend()
    plt.grid(True)
    plt.show()
