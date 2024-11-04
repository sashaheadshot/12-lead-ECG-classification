import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from scipy.io import loadmat
import os

# Set device for GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ECGDataset(Dataset):
    def __init__(self, directory, labels_df):
        self.directory = directory
        self.labels_df = labels_df

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        filename = self.labels_df.iloc[idx]['Filename'] + '.mat'
        filepath = os.path.join(self.directory, filename)
        data = loadmat(filepath)['val']
        data = data / np.max(np.abs(data)) if np.max(np.abs(data)) != 0 else data # Normalizing the data

        numeric_label = self.labels_df.iloc[idx][
            ['Class_0', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5']].values
        label_indices = [i for i, val in enumerate(numeric_label) if val == 1]  # Here we loop through each  value to check if the class is 0 or 1

        # Convert to binary indicator matrix
        y_binary = np.zeros(num_classes, dtype=int)
        for label in label_indices:
            y_binary[label] = 1

        return torch.tensor(data, dtype=torch.float32), torch.tensor(y_binary, dtype=torch.float32) # Converting data to tensors

# labels
labels_file_path = 'E:\\Coding\\Jupyter_files\\ECG_2\\Features\\Labels.csv'
labels_df = pd.read_csv(labels_file_path)

# Data
directory = 'E:\\Coding\\Jupyter_files\\ECG_2\\Data_saves\\Final_data'
dataset = ECGDataset(directory, labels_df)


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class SimplifiedVGG16_1D(nn.Module):
    def __init__(self, num_classes, feature_size=256):
        super(SimplifiedVGG16_1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.fc1 = nn.Linear(256 * (5000 // 16), 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc_features = nn.Linear(4096, feature_size)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        features = self.fc_features(x)
        logits = self.fc3(x)
        return logits, features


num_classes = len(labels_df.columns) - 1
model = SimplifiedVGG16_1D(num_classes=num_classes).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)


num_epochs = 40
train_losses = []
val_losses = []
best_loss = float('inf')
all_test_preds = []
all_test_labels = []
epochs_no_improve = 0
patience = 5
clip_value = 1.0

print("Training is in progress...")

# Training loop
for epoch in range(num_epochs):

    all_train_extracted_features = []  # Saving features outside the loop, otherwise they will be saved for each epoch and concatenated togehter.
    all_test_extracted_features = []

    model.train()
    running_loss = 0.0

    # Training phase
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()  # Why zero grad ?
        outputs, features = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_value)  # Gradient clipping
        optimizer.step()
        running_loss += loss.item()

        # Append extracted features for the training set
        all_train_extracted_features.append(features.cpu().detach().numpy())

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, features = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Append extracted features for the testing set
            all_test_extracted_features.append(features.cpu().detach().numpy())
            all_test_preds.append(torch.sigmoid(outputs).cpu().numpy())
            all_test_labels.append(labels.cpu().numpy())

    val_loss /= len(test_loader)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

    # Early stopping logic
    if val_loss < best_loss:
        best_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print("Early stopping triggered")
        break

# Concatatenate test and train features
train_extracted_features_array = np.concatenate(all_train_extracted_features, axis=0)
test_extracted_features_array = np.concatenate(all_test_extracted_features, axis=0)

# Print shapes of extracted features
print(f"Extracted training features shape: {train_extracted_features_array.shape}")
print(f"Extracted testing features shape: {test_extracted_features_array.shape}")


combined_extracted_features_array = np.concatenate((train_extracted_features_array, test_extracted_features_array), axis=0)  # Concatatenate test and train features together

combined_features_save_path = 'C:\\Users\\Alex\\Desktop\\combined_extracted_features.csv'
pd.DataFrame(combined_extracted_features_array).to_csv(combined_features_save_path, index=False)
print(f"Final combined extracted features shape: {combined_extracted_features_array.shape}")

# Compute metrics
all_test_preds = np.concatenate(all_test_preds, axis=0)
all_test_labels = np.concatenate(all_test_labels, axis=0)

# Convert predictions to binary
predicted_labels = (all_test_preds > 0.5).astype(int)


test_f1 = f1_score(all_test_labels, predicted_labels, average='micro')
test_precision = precision_score(all_test_labels, predicted_labels, average='micro')
test_recall = recall_score(all_test_labels, predicted_labels, average='micro')
test_accuracy = (predicted_labels == all_test_labels).mean()

print(f'Test F1 Score: {test_f1:.4f}')
print(f'Test Precision: {test_precision:.4f}')
print(f'Test Recall: {test_recall:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Debug statements
print(f"Shape of labels DataFrame: {labels_df.shape}")
print(f"Total files processed: {len(dataset)}")
