import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, hamming_loss, precision_score, recall_score
import torch.nn.functional as F
import matplotlib.pyplot as plt



labels_path = 'E:\Coding\Jupyter_files\ECG_2\Features\Labels.csv'
resnet_features_path = "E:\\Coding\\ECG\\Features\\resnet_features.csv"
qrstp_features_path = 'E:\Coding\Jupyter_files\ECG_2\Features\QRSTP_features.json'
rr_features_path = 'E:\Coding\Jupyter_files\ECG_2\Features\RR_features.json'




with open(qrstp_features_path, 'r') as f:
    qrstp_features = json.load(f)
qrstp_features = {key.split('\\')[-1]: value for key, value in qrstp_features.items()}

with open(rr_features_path, 'r') as f:
    other_features = json.load(f)
other_features = {key.split('\\')[-1]: value for key, value in other_features.items()}





resnet_features = pd.read_csv(resnet_features_path)
labels_df = pd.read_csv(labels_path)





hc_features = []
labels_list = []



for _, row in labels_df.iterrows():
    filename = row['Filename'] + '.mat'
    label = row[['Class_0', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5']].values.tolist()


    if filename in qrstp_features and filename in other_features:
        try:
            qrstp_feature = qrstp_features[filename]
            r_peaks = np.array(qrstp_feature.get('ECG_R_Peaks', []))[:8]
            p_peaks = np.array(qrstp_feature.get('ECG_P_Peaks', []))[:8]
            t_peaks = np.array(qrstp_feature.get('ECG_T_Peaks', []))[:8]
            q_peaks = np.array(qrstp_feature.get('ECG_Q_Peaks', []))[:8]
            s_peaks = np.array(qrstp_feature.get('ECG_S_Peaks', []))[:8]

            other_feature = other_features[filename]
            rr_intervals = np.array(other_feature.get('RR Intervals: ', []))[:8]
            rr_interval_median = np.array(other_feature.get('RR Interval Median: ', [])).flatten()
            hr_values = np.array(other_feature.get('HR: ', []))[:8]
            avg_hr = np.array(other_feature.get('AVG HR: ', [])).flatten()
            sdnn = np.array(other_feature.get('SDNN: ', [])).flatten()
            rmssd = np.array(other_feature.get('RMSSD: ', [])).flatten()
            pnn60 = np.array(other_feature.get('PNN60: ', [])).flatten()

            combined_feature = []
            combined_feature.extend(r_peaks)   
            combined_feature.extend(p_peaks)
            combined_feature.extend(t_peaks)
            combined_feature.extend(q_peaks)
            combined_feature.extend(s_peaks)   

            combined_feature.extend(rr_intervals)
            combined_feature.extend(rr_interval_median)
            combined_feature.extend(hr_values)
            combined_feature.extend(avg_hr)
            combined_feature.extend(sdnn)
            combined_feature.extend(rmssd)
            combined_feature.extend(pnn60)

            hc_features.append(combined_feature)
            labels_list.append(label)


        except Exception as e:
            print(f"Error processing file {filename}: {e}")





array_features = np.array(hc_features)

min_values = np.min(array_features)
max_values = np.max(array_features)
scaled_hc_features = 2 * (array_features - min_values) / (max_values - min_values) - 1






array_resnet_features = np.array(resnet_features)

print("Array Features shape: ",array_resnet_features.shape)
print("Array Features: ",array_resnet_features)

combined_features = []



for i in range(len(labels_df)):
    label_filename = labels_df.iloc[i, 0] + '.mat'

    matched_row = None
    for j in range(len(array_resnet_features)):
        if array_resnet_features[j, 0] == label_filename:  
            matched_row = array_resnet_features[j, 1:] 
            break

    if matched_row is not None:

        min_values = np.min(matched_row)
        max_values = np.max(matched_row)
        normalized_matched_row = 2 * (matched_row - min_values) / (max_values - min_values) - 1

        # Get the handcrafted feature vector at the same index

        hc_row = scaled_hc_features[i]

        combined_row = np.concatenate((normalized_matched_row, hc_row))
        #combined_row = normalized_matched_row



        combined_features.append(combined_row)
    else:
        print(f"Warning: Filename {label_filename} not found in DL features.")

combined_features_array = np.array(combined_features, dtype=np.float32)

print(combined_features_array.shape)


tensor_features = torch.tensor(combined_features_array, dtype=torch.float32)
tensor_labels = torch.tensor(labels_list, dtype=torch.float32)

print("tensor shape: ", tensor_features.shape)
print("Tensor features: ",tensor_features)
print('Tensor labels: ',tensor_labels)




class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.3)

        self.output = nn.Linear(64, num_classes)

    def forward(self, x):

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        x = self.output(x)

        return x





X_train, X_val, y_train, y_val = train_test_split(tensor_features, tensor_labels, test_size=0.2, random_state=42)


train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


num_epochs = 400
input_size = X_train.shape[1]
num_classes = y_train.shape[1]
model = SimpleNN(input_size=input_size, num_classes=num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
loss_function = nn.BCEWithLogitsLoss()




all_preds = []
all_labels = []

train_losses = []
val_losses = []


for epoch in range(num_epochs):
    model.train() 
    running_train_loss = 0.0
    running_train_hamming_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Clear previous gradients

        outputs = model(inputs)
        train_loss = loss_function(outputs, labels)
        running_train_loss += train_loss.item()
        train_hamming_loss = hamming_loss(labels.cpu().numpy(), torch.sigmoid(outputs).cpu().detach().numpy() > 0.5)
        running_train_hamming_loss += train_hamming_loss

        train_loss.backward()
        optimizer.step()



    avg_train_loss = running_train_loss / len(train_loader)
    avg_train_hamming_loss = running_train_hamming_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()  
    running_val_loss = 0.0
    running_val_hamming_loss = 0.0

    with torch.no_grad():  # No gradients needed during validation
        for inputs, labels in val_loader:
            outputs = model(inputs)


            val_loss = loss_function(outputs, labels)
            running_val_loss += val_loss.item()

            preds = torch.sigmoid(outputs).cpu().detach().numpy() > 0.5  # Convert to binary (0 or 1)
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

            val_hamming_loss = hamming_loss(labels.cpu().numpy(), preds)
            running_val_hamming_loss += val_hamming_loss

    avg_val_loss = running_val_loss / len(val_loader)
    avg_val_hamming_loss = running_val_hamming_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {avg_train_loss:.4f}, Train Hamming Loss: {avg_train_hamming_loss:.4f}, '
          f'Validation Loss: {avg_val_loss:.4f}, Validation Hamming Loss: {avg_val_hamming_loss:.4f}')


all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)


accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='micro')
recall = recall_score(all_labels, all_preds, average='micro')
f1 = f1_score(all_labels, all_preds, average='micro')

# Print final results
print(f'Final Results after Training:')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')



'''
Classification Report
'''

val_preds = []
val_labels = []


with torch.no_grad():
    for batch in val_loader:
        inputs, labels = batch
        inputs, labels = inputs, labels

        # Forward pass
        outputs = model(inputs)
        outputs = (outputs > 0.5).float()  # Apply threshold (0.5)

        val_preds.append(outputs)
        val_labels.append(labels)

val_preds = torch.cat(val_preds, dim=0).cpu().numpy()
val_labels = torch.cat(val_labels, dim=0).cpu().numpy()

print("\nValidation Classification Report:")
print(classification_report(val_labels, val_preds))


plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', color='blue')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Deep learning - hand crafted features Training and Validation loss')
plt.legend()
plt.grid(True)
plt.savefig('dl_hc_cnn_plot.png', dpi=300)
plt.show()


#Metrics
val_accuracy = accuracy_score(val_labels, val_preds)
val_precision = precision_score(val_labels, val_preds, average='macro')
val_recall = recall_score(val_labels, val_preds, average='macro')
val_f1 = f1_score(val_labels, val_preds, average='macro')
