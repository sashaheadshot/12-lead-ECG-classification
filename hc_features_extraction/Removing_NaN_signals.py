import neurokit2 as nk
import json
import numpy as np
from tqdm import tqdm
from data_loader_mat import load_data

#EXTRACTION AND REMOVING NAN AND UPPER LIMIT INDECIES

def clean_peaks(peaks):
    return [int(peak) for peak in peaks if not np.isnan(peak) and peak < 5000]


directory = r'E:\Coding\Jupyter_files\ECG_2\Data_saves\Final_data'
data = load_data(directory)

ecg_features = {}
removed_files = []
progress_bar = tqdm(total=len(data), desc='Extracting ECG features')

for filepath, signals in data.items():
    ecg_signal = signals['val'][0]

    # Check for valid signal
    if len(ecg_signal) != 5000 or np.isnan(ecg_signal).any():
        print(f"Invalid ECG signal in file {filepath}: Length = {len(ecg_signal)}, Contains NaN.")
        removed_files.append(filepath)
        continue

    try:
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

        # Check if any peaks exceed the threshold
        peaks_lists = [p_peaks_clean, r_peaks_clean, t_peaks_clean, q_peaks_clean, s_peaks_clean]
        if any(any(peak >= 5000 for peak in peaks) for peaks in peaks_lists):
            print(f"Some peaks exceeded the threshold in file {filepath}.")
            removed_files.append(filepath)
            continue

        # Store features
        ecg_features[filepath] = {
            "ECG_R_Peaks": r_peaks_clean,
            "ECG_P_Peaks": p_peaks_clean,
            "ECG_T_Peaks": t_peaks_clean,
            "ECG_Q_Peaks": q_peaks_clean,
            "ECG_S_Peaks": s_peaks_clean
        }

    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        removed_files.append(filepath)
        continue

    progress_bar.update(1)

progress_bar.close()



