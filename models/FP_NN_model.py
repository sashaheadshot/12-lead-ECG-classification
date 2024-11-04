import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


# Warning: Sorry this code is so shit.


# Loading features and labels
labels_path = 'E:\Coding\Jupyter_files\ECG_2\Features\Labels.csv'
labels_df = pd.read_csv(labels_path)

dl_features_filepath = 'E:\Coding\Jupyter_files\ECG_2\Features\dl_features.csv'
dl_features_df = pd.read_csv(dl_features_filepath)


# Using double \\ to separate the file name and the path.
qrstp_features_path = 'E:\Coding\Jupyter_files\ECG_2\Features\QRSTP_features.json'
with open(qrstp_features_path, 'r') as file:
    qrstp_features = json.load(file)
qrstp_features = {key.split('\\')[-1]: value for key, value in qrstp_features.items()}

rr_features_path = 'E:\Coding\Jupyter_files\ECG_2\Features\RR_features.json'
with open(rr_features_path, 'r') as file:
    rr_features = json.load(file)
rr_features = {key.split('\\')[-1]: value for key, value in rr_features.items()}

# Preprocessing
all_features = []
all_y_binaries = []
num_classes = 6

# Changing hand-crafted features to np array, setting theshold to 8 (8 numbers per feature except for scalars)
for idx in range(len(labels_df)):
    numeric_label = labels_df.iloc[idx][['Class_0', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5']].values
    label_indices = [i for i, val in enumerate(numeric_label) if val == 1]
    y_binary = np.zeros(num_classes, dtype=int)
    for label in label_indices:
        y_binary[label] = 1
    all_y_binaries.append(y_binary)

    filename = labels_df.iloc[idx]["Filename"]
    if f"{filename}.mat" in qrstp_features and f"{filename}.mat" in rr_features:
        qrstp_feature = qrstp_features[f"{filename}.mat"]
        other_feature = rr_features[f"{filename}.mat"]

        r_peaks = np.array(qrstp_feature['ECG_R_Peaks'])[:8]
        p_peaks = np.array(qrstp_feature['ECG_P_Peaks'])[:8]
        t_peaks = np.array(qrstp_feature['ECG_T_Peaks'])[:8]
        q_peaks = np.array(qrstp_feature['ECG_Q_Peaks'])[:8]
        s_peaks = np.array(qrstp_feature['ECG_S_Peaks'])[:8]
        rr_intervals = np.array(other_feature['RR Intervals: '])[:8]

        rr_interval_median = other_feature['RR Interval Median: ']
        avg_hr = other_feature['AVG HR: ']
        sdnn = other_feature['SDNN: ']
        rmssd = other_feature['RMSSD: ']
        pnn60 = other_feature['PNN60: ']

        combined_features = np.concatenate((r_peaks, p_peaks, t_peaks, q_peaks, s_peaks, rr_intervals,
                                            np.array([rr_interval_median]), np.array([avg_hr]),
                                            np.array([sdnn]), np.array([rmssd]), np.array([pnn60])))
        dl_features = dl_features_df.iloc[idx].values # <-----------Looping through DL extracted features of each sample.
        final_combined_features = np.concatenate((combined_features, dl_features))
        all_features.append(final_combined_features)

# Convert to NumPy arrays
y_binaries_array = np.array(all_y_binaries)
combined_features_array = np.array(all_features)

# Normalize features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(combined_features_array)

# Combine features and labels
final_dataset = np.concatenate((X_normalized, y_binaries_array), axis=1)

# Model
X = final_dataset[:, :-num_classes]  # Features
y = final_dataset[:, -num_classes:]   # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_inputs = torch.tensor(X_train, dtype=torch.float32)
train_labels = torch.tensor(y_train, dtype=torch.float32)
test_inputs = torch.tensor(X_test, dtype=torch.float32)
test_labels = torch.tensor(y_test, dtype=torch.float32)

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

# Initialize model, loss function, and optimizer
input_size = X_normalized.shape[1]
model = SimpleNN(input_size, num_classes)


# Probably a clipping gradient is needed here

# Calculate class weights
class_counts = np.bincount(y_train.argmax(axis=1))
class_weights = 1.0 / class_counts  # Not sure which value to choose
class_weights = class_weights / np.sum(class_weights)  # Normalize weights
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32)) # I assume using cross entropy is the only way for multiple classes.
optimizer = optim.Adam(model.parameters(), lr=0.00003)

# Training
num_epochs = 200
batch_size = 128
clipping_value = 1.0
epochs_no_improvement = 0
patience = 40
best_loss = float('inf')


for epoch in range(num_epochs):
    model.train()
    indices = np.arange(len(train_inputs))
    np.random.shuffle(indices)

    for start_idx in range(0, len(train_inputs), batch_size):
        end_idx = start_idx + batch_size
        batch_indices = indices[start_idx:end_idx]

        batch_inputs = train_inputs[batch_indices]
        batch_labels = train_labels[batch_indices]

        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels.argmax(axis=1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clipping_value)  # Gradient clipping
        optimizer.step()

    # Evaluation
    model.eval()
    val_loss = 0.0
    num_batches = 0

    with torch.no_grad(): # Disabling gradient calculation
        for start_idx in range(0, len(test_inputs), batch_size):
            end_idx = start_idx + batch_size

            # Extracting current batch
            batch_inputs = test_inputs[start_idx:end_idx]
            batch_labels = test_labels[start_idx:end_idx]

            test_outputs = model(batch_inputs)
            batch_loss = criterion(test_outputs, batch_labels.argmax(axis=1))
            val_loss += batch_loss.item()
            num_batches += 1

        val_loss /= num_batches  # Average loss over all batches

    # Print losses every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

# Early stopping logic
    if val_loss < best_loss:
        best_loss = val_loss
        epochs_no_improvement = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        epochs_no_improvement += 1

    if epochs_no_improvement >= patience:
        print("Early stopping triggered")
        break

# Final evaluation
with torch.no_grad():
    final_outputs = model(test_inputs)
    predicted_classes = final_outputs.argmax(axis=1).numpy()
    true_classes = test_labels.argmax(axis=1).numpy()

    accuracy = accuracy_score(true_classes, predicted_classes)
    f1 = f1_score(true_classes, predicted_classes, average='weighted')

    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test F1 Score: {f1:.4f}')
    print('Classification Report:')
    print(classification_report(true_classes, predicted_classes, zero_division=0))
