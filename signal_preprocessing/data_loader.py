import os
import scipy.io
from tqdm import tqdm

def load_data(main_directory):
    total_files = sum(len(files) for _, _, files in os.walk(main_directory))
    progress_bar = tqdm(total=total_files, desc="Data is loading...")
    loaded_data = {}

    for root, dirs, files in os.walk(main_directory):
        for filename in files:
            if filename.endswith('.mat') or filename.endswith('.hea'):
                file_path = os.path.join(root, filename)
                if filename.endswith('.mat'):
                    data = scipy.io.loadmat(file_path)
                elif filename.endswith('.hea'):
                    with open(file_path, 'r') as f:
                        data = f.read()
                loaded_data[file_path] = data
            progress_bar.update(1)
    progress_bar.close()

    return loaded_data
