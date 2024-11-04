import neurokit2 as nk
import json
import numpy as np
from tqdm import tqdm
from data_loader_mat import load_data


def clean_peaks(peaks):
    return [int(peak) for peak in peaks if not np.isnan(peak) and peak < 5000]


directory = r'E:\Coding\Jupyter_files\ECG_2\Data_saves\Final_data'
data = load_data(directory)

ecg_features = {}
progress_bar = tqdm(total=len(data), desc='Extracting ECG features')

for filepath, signals in data.items():
    ecg_signal = signals['val'][0]


    _, waves_peak = nk.ecg_delineate(ecg_signal, sampling_rate=500, method="peak")
    _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=500)

    r_peaks = rpeaks['ECG_R_Peaks']
    p_peaks = waves_peak['ECG_P_Peaks']
    t_peaks = waves_peak['ECG_T_Peaks']
    q_peaks = waves_peak['ECG_Q_Peaks']
    s_peaks = waves_peak['ECG_S_Peaks']

    # Clean the peaks
    p_peaks_clean = clean_peaks(p_peaks)
    r_peaks_clean = clean_peaks(r_peaks)
    t_peaks_clean = clean_peaks(t_peaks)
    q_peaks_clean = clean_peaks(q_peaks)
    s_peaks_clean = clean_peaks(s_peaks)


    ecg_features[filepath] = {
        "ECG_R_Peaks": r_peaks_clean,
        "ECG_P_Peaks": p_peaks_clean,
        "ECG_T_Peaks": t_peaks_clean,
        "ECG_Q_Peaks": q_peaks_clean,
        "ECG_S_Peaks": s_peaks_clean
    }





    progress_bar.update(1)

progress_bar.close()

output_json_path = 'C:\\Users\\Alex\\Desktop\\QRSTP_features.json'
with open(output_json_path, 'w') as json_file:
    json.dump(ecg_features, json_file)

