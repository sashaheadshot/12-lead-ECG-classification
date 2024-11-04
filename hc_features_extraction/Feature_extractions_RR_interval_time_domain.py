import neurokit2 as nk
import json
import numpy as np
from data_loader_mat import load_data
from tqdm import tqdm


directory = r'E:\Coding\Jupyter_files\ECG_2\Data_saves\Final_data'
data = load_data(directory)

ecg_RR_features = {}
progress_bar = tqdm(total=len(data), desc='Extracting ECG features')

for filepath, signals in data.items():
    ecg_signal = signals['val'][0]

    #_, waves_peak = nk.ecg_delineate(ecg_signal, sampling_rate=500, method="peak")
    _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=500)
    r_peaks = rpeaks['ECG_R_Peaks']

    # RR_intervals/RR_interval_median

    rr_intervals_samples = np.diff(r_peaks)
    sampling_rate = 500
    rr_intervals_time = rr_intervals_samples / sampling_rate
    rr_median = np.median(rr_intervals_time)

    # HR

    hr = 60 / rr_intervals_time
    hr_median = np.median(hr)
    rounded_hr_median = round(hr_median)

    # HRV

    sdnn = np.std(rr_intervals_time)
    sdnn_ms = sdnn * 1000
    rounded_sdnn = round(sdnn_ms)


    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals_time))))
    rmssd_ms = rmssd * 1000
    rounded_rmssd = round(rmssd_ms)

    # PNN60

    rr_diff = np.abs(np.diff(rr_intervals_time))
    pnn60_count = np.sum(rr_diff > 0.06)
    total_intervals = len(rr_intervals_time) - 1
    pnn60 = (pnn60_count / total_intervals) * 100 if total_intervals > 0 else 0
    rounded_pnn60 = round(pnn60)

    ecg_RR_features[filepath] = {
        'RR Intervals: ': rr_intervals_time.tolist(),
        'RR Interval Median: ': rr_median,
        'HR: ':np.round(hr).tolist(),
        'AVG HR: ':rounded_hr_median,
        'SDNN: ':rounded_sdnn,
        'RMSSD: ':rounded_rmssd,
        'PNN60: ':rounded_pnn60
    }
    progress_bar.update(1)
progress_bar.close()

output_json_path = 'C:\\Users\\Alex\\Desktop\\RR_features_TEST.json'
with open(output_json_path, 'w') as json_file:
    json.dump(ecg_RR_features, json_file)
