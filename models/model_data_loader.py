import scipy.io as sio
import os
from tqdm import tqdm

def model_data_loader(directory):
    mat_files = os.listdir(directory)
    progres_bar = tqdm(total=len(mat_files), desc='Loading data')
    ecg_data = {}
    for filename in os.listdir(directory):
        if filename.endswith('.mat'):
            mat_file = os.path.join(directory,filename)
            data = sio.loadmat(mat_file)
            ecg_data[filename] = data['val']
            progres_bar.update(1)
    progres_bar.close()
    return ecg_data