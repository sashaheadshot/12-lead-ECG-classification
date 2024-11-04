import pandas as pd

def labels_loader(csv_file):
    labels_df = pd.read_csv(csv_file)
    return labels_df