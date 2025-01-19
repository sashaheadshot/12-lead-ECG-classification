import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from scipy.io import loadmat
import os
# Set device for GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# File paths
labels_file_path = 'E:\\Coding\\Jupyter_files\\ECG_2\\Features\\Labels.csv'
labels_df = pd.read_csv(labels_file_path)
directory = 'E:\\Coding\\Jupyter_files\\ECG_2\\Data_saves\\Final_data'


# ECG Dataset class
class ECGDataset(Dataset):
    def __init__(self, directory, labels_df):
        self.directory = directory
        self.labels_df = labels_df
        self.labels_list = []

        # Store labels as lists
        for idx, row in labels_df.iterrows():
            self.labels_list.append(row[['Class_0', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5']].tolist())

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, idx):
        filename = self.labels_df.iloc[idx]['Filename'] + '.mat'
        filepath = os.path.join(self.directory, filename)
        data = loadmat(filepath)['val']
        data = data / np.max(np.abs(data)) if np.max(np.abs(data)) != 0 else data  # Normalizing the data

        # Get the label
        label = self.labels_list[idx]

        # Ensure data is in the shape (1, 12, 5000) for 1D ConvNet
        data = np.expand_dims(data, axis=0)  # Add channel dimension
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), filename


dataset = ECGDataset(directory, labels_df)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Residual Block for 1D CNN
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


# ResNet-1D Model
class ResNet1D(nn.Module):
    def __init__(self, num_classes=6):
        super(ResNet1D, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 12 input channels for 12 leads
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        layers = []
        layers.append(ResidualBlock1D(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.squeeze(1)  # This removes the second dimension (the '1' channel dimension)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        features = self.avgpool(x)
        features = torch.flatten(features, 1)
        output = self.fc(features)
        return output, features  # Return both the output and the extracted features



# Model initialization
model = ResNet1D(num_classes=6).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# Loss function
criterion = nn.BCEWithLogitsLoss()


# Number of samples should match the length of your dataset
num_train_samples = len(train_loader.dataset)
num_val_samples = len(test_loader.dataset)


print(num_train_samples, num_val_samples)

# Resetting the feature and filename lists for each epoch
train_features = []  # To store the features from the training set
train_filenames = []  # To store the filenames from the training set

val_features = []  # To store the features from the validation set
val_filenames = []  # To store the filenames from the validation set

# Training loop
num_epochs = 15
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    all_train_preds = []
    all_train_labels = []

    # Initialize the lists at the beginning of each epoch
    epoch_train_features = []  # To accumulate features for this epoch
    epoch_train_filenames = []  # To accumulate filenames for this epoch

    for inputs, labels, filenames in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs, features = model(inputs)  # Get the output and features

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        all_train_preds.append(outputs.sigmoid().cpu().detach().numpy())
        all_train_labels.append(labels.cpu().detach().numpy())

        # Append features for this batch (only for the current batch)
        epoch_train_features.append(features.cpu().detach().numpy())
        epoch_train_filenames.extend(filenames)  # Save filenames for this batch

    train_loss /= len(train_loader)
    all_train_preds = np.concatenate(all_train_preds, axis=0)
    all_train_labels = np.concatenate(all_train_labels, axis=0)

    train_preds_binary = (all_train_preds > 0.5).astype(int)
    train_accuracy = accuracy_score(all_train_labels, train_preds_binary)
    train_precision = precision_score(all_train_labels, train_preds_binary, average='weighted', zero_division=1)
    train_recall = recall_score(all_train_labels, train_preds_binary, average='weighted', zero_division=1)
    train_f1 = f1_score(all_train_labels, train_preds_binary, average='weighted')

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
          f"Train Accuracy: {train_accuracy:.4f}, Train Precision: {train_precision:.4f}, "
          f"Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}")

    train_losses.append(train_loss)

    # Validation phase
    model.eval()
    val_loss = 0.0
    all_val_preds = []
    all_val_labels = []

    # Initialize the lists at the beginning of the validation phase
    epoch_val_features = []  # To accumulate features for this validation
    epoch_val_filenames = []  # To accumulate filenames for this validation

    with torch.no_grad():
        for inputs, labels, filenames in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, features = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            all_val_preds.append(outputs.sigmoid().cpu().detach().numpy())
            all_val_labels.append(labels.cpu().detach().numpy())

            # Append features for this validation batch
            epoch_val_features.append(features.cpu().detach().numpy())
            epoch_val_filenames.extend(filenames)  # Save filenames for this validation batch

    val_loss /= len(test_loader)
    all_val_preds = np.concatenate(all_val_preds, axis=0)
    all_val_labels = np.concatenate(all_val_labels, axis=0)

    val_preds_binary = (all_val_preds > 0.5).astype(int)
    val_accuracy = accuracy_score(all_val_labels, val_preds_binary)
    val_precision = precision_score(all_val_labels, val_preds_binary, average='weighted', zero_division=1)
    val_recall = recall_score(all_val_labels, val_preds_binary, average='weighted', zero_division=1)
    val_f1 = f1_score(all_val_labels, val_preds_binary, average='weighted')

    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, "
          f"Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}, "
          f"Validation F1: {val_f1:.4f}")

    val_losses.append(val_loss)

    # Save features at the end of the last epoch
    if epoch == num_epochs - 1:  # Save at the last epoch
        # Concatenate features for the training and validation datasets
        all_train_features = np.concatenate(epoch_train_features, axis=0)
        all_val_features = np.concatenate(epoch_val_features, axis=0)

        # Concatenate filenames
        all_train_filenames = np.array(epoch_train_filenames)
        all_val_filenames = np.array(epoch_val_filenames)

        # Save combined features and filenames
        all_features = np.concatenate([all_train_features, all_val_features], axis=0)
        all_filenames = np.concatenate([all_train_filenames, all_val_filenames], axis=0)

        # Ensure that the number of features matches the number of filenames
        assert len(all_features) == len(all_filenames), "Number of features and filenames mismatch"

        # Create a DataFrame and save to CSV
        features_df = pd.DataFrame(all_features)
        features_df['filename'] = all_filenames  # Add the filenames column
        features_df = features_df[['filename'] + [col for col in features_df.columns if col != 'filename']]
        features_df.to_csv('resnet_all_features_with_filenames.csv', index=False)

    # Learning rate scheduler
    scheduler.step(val_loss)






# Save the model (the best model from validation loss)
#torch.save(model.state_dict(), 'best_model.pth')
#print("Best model saved.")
