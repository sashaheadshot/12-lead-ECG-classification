import os
import scipy.io
from tqdm import tqdm

def data_saver_v1(processed_data, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    
    progress_bar = tqdm(total=len(processed_data), desc="Saving files...")

    for file_name, data in processed_data.items():
        output_file_path = os.path.join(output_directory, os.path.basename(file_name))
        if file_name.endswith('.hea'):
            with open(output_file_path, 'w') as f:
                f.write(data)
        elif file_name.endswith('.mat'):
            data_dict = {'val': data['val']}  # Assuming 'val' is the key containing the numpy array
            scipy.io.savemat(output_file_path, data_dict)
        
        progress_bar.update(1)

    progress_bar.close()
    print("Data saved to directory successfully.")
