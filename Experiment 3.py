import os
import re
import numpy as np
import pandas as pd
import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
#from pythonProject7.podgotovkaPandas import X_train
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking

def parse_value(x):
    if isinstance(x, str):
        try:
            # Try to parse string as a list (e.g., "[1, 2, 3]" -> [1, 2, 3])
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            # If it's not a valid list, treat it as a single value
            return x
    return x

def parse_custom_array(s):
    # Remove np.float32() wrappers while keeping the numeric values
    cleaned = re.sub(r'np\.float32\(([^)]+)\)', r'\1', s)
    # Convert to actual list
    data = ast.literal_eval(cleaned)
    # Convert numbers to float32 if needed
    return [[x[0]] + [np.float32(y) for y in x[1:]] for x in data]

def get_mask_subdirs_os2(directory_path):
    if not os.path.isdir(directory_path):
        raise ValueError(f"'{directory_path}' is not a valid directory")

    video_pattern = re.compile(r'^videos([1-2]?[0-9])$')
    mask_pattern = re.compile(r'^mask\d+$')

    mask_subdirs = []

    for video_dir in os.listdir(directory_path):
        video_path = os.path.join(directory_path, video_dir)
        if os.path.isdir(video_path) and video_pattern.match(video_dir):
            for subdir in os.listdir(video_path):
                subdir_path = os.path.join(video_path, subdir)
                if os.path.isdir(subdir_path) and mask_pattern.match(subdir):
                    mask_subdirs.append(subdir_path)

    return sorted(mask_subdirs)
def getData():
    xTrain_combined = []
    yTrain_combined = []
    for mask in get_mask_subdirs_os2(f'C:/Users/Alexs/PyCharmMiscProject/pythonProject7/object_tracking/object_tracking/samMooooolq/segmentAnything'):
        #if i > 0:
            #continue
        #i = i + 1
        print(mask)
        df = pd.read_csv(mask + '/trainData.csv')
        capData = []
        for row in df['cap']:
            row = row.replace("np.float32", "float")
            #row = re.sub(r"float\(([-0-9.]+)\)", r"\1", row)
            row = re.sub(r"float\((-?\d*\.?\d+(?:e[-+]?\d+)?)\)", r"\1", row)
            try:
                # Parse the string into a Python list
                parsed_data = ast.literal_eval(row)

                # Convert float values to np.float32
                result = [[item[0], np.float32(item[1]), np.float32(item[2]), np.float32(item[3]), np.float32(item[4])]
                          for item in parsed_data]

            except ValueError as e:
                print(f"Error parsing string: {e}")
                print(f"Problematic string: {row}")
                raise
            print(result)
            cap = []
            index = 1
            for point in result:
                while True:
                    if point[0] == index:
                        cap.append(point[1:])
                        index += 1
                        break
                    elif point[0] != index:
                        cap.append([0, 0, 0, 0])
                        index += 1
                        if index == 10:
                            break

            while index < 10:
                cap.append([0, 0, 0, 0])
                index += 1
            flat_list = [item for sublist in cap for item in sublist]
            print(flat_list, len(flat_list))
            capData.append(flat_list)

        centerSizeData = []
        for row in df['coordinates']:
            row = row.replace("np.float32", "float")
            # row = re.sub(r"float\(([-0-9.]+)\)", r"\1", row)
            row = re.sub(r"float\((-?\d*\.?\d+(?:e[-+]?\d+)?)\)", r"\1", row)
            print(row)
            try:
                # Parse the string into a Python list
                parsed_data = ast.literal_eval(row)

                # Convert float values to np.float32]

            except ValueError as e:
                print(f"Error parsing string: {e}")
                print(f"Problematic string: {row}")
                raise
            size = parsed_data[2] * parsed_data[3]
            centerX = parsed_data[0] + (parsed_data[2] / 2)
            centerY = parsed_data[1] + (parsed_data[3] / 2)
            print(size, centerX, centerY)
            centerSizeData.append([np.float32(size), np.float32(centerX), np.float32(centerY)])


        scoreData = []
        for row in df['score']:
            print(row)
            scoreData.append(row)

        combined = []
        for l1, l2 in zip(capData, centerSizeData):
            combined.append(l1 + l2)
        print(combined[0], len(combined[0]))
        X_array = np.array(combined)
        print(X_array.shape)
        y_array = np.array(scoreData)
        print(y_array.shape)

        #dataEncoded = encode_pairs(dataset)
        xTrain_combined.append(X_array)
        yTrain_combined.append(y_array)


        # Verify shapes
    print("xTrain_final shape:", xTrain_combined[0].shape)
    print("yTrain_final shape:", yTrain_combined[0].shape)



    return xTrain_combined, yTrain_combined

X, y = getData()
print(X[0].shape, len(X))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# =============================================
# 1. Data Preparation
# =============================================

class SequenceDataset(Dataset):
    def __init__(self, sequences, targets, window_size=10):
        """
        Custom dataset for sequence prediction

        Args:
            sequences: List of numpy arrays (each array is [timesteps, features])
            targets: List of numpy arrays with target values (aligned with sequences)
            window_size: Number of consecutive timesteps to use for prediction
        """
        self.window_size = window_size
        self.features = sequences[0].shape[1]  # Number of features (39)

        # Process all sequences into windows
        self.X = []
        self.y = []

        for seq_idx, (seq, target_seq) in enumerate(zip(sequences, targets)):
            seq_length = seq.shape[0]

            # Skip sequences shorter than window size
            if seq_length < window_size:
                continue

            # Create windows
            for i in range(seq_length - window_size + 1):
                window = seq[i:i + window_size]
                target = target_seq[i + window_size - 1]  # Target is the first element in window

                self.X.append(window)
                self.y.append(target)

        # Convert to numpy arrays
        self.X = np.stack(self.X)
        self.y = np.array(self.y).reshape(-1, 1)

        # Normalize features
        self.scaler_X = StandardScaler()
        original_shape = self.X.shape
        self.X = self.scaler_X.fit_transform(
            self.X.reshape(-1, self.features)).reshape(original_shape)

        # Normalize targets
        self.scaler_y = StandardScaler()
        self.y = self.scaler_y.fit_transform(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert to PyTorch tensors
        x = torch.FloatTensor(self.X[idx])
        y = torch.FloatTensor(self.y[idx])
        return x, y


# =============================================
# 2. Model Definition (same as before)
# =============================================

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Using GRU for better performance with long sequences
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward pass through GRU
        out, _ = self.gru(x, h0)

        # We only want the output from the last timestep
        out = out[:, -1, :]

        # Fully connected layer
        out = self.fc(out)
        return out


# =============================================
# 3. Training Setup (same as before)
# =============================================

def train_model(dataset, epochs=100, batch_size=32, learning_rate=0.001):
    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    input_size = dataset.features
    hidden_size = 64
    num_layers = 2
    output_size = 1

    model = RNNModel(input_size, hidden_size, num_layers, output_size)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        epoch_train_loss = 0

        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        # Calculate validation loss
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                epoch_val_loss += loss.item()

        # Save losses for plotting
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return model, dataset.scaler_y

window_size = 10
dataset = SequenceDataset(X, y, window_size=window_size)
print("Shapea", dataset.X.shape)

# 3. Train model
model, target_scaler = train_model(dataset, epochs=50, batch_size=32, learning_rate=0.001)

# 4. Example prediction (using the first window from validation set)
val_loader = DataLoader(dataset, batch_size=1, shuffle=True)
sample_X, sample_y = next(iter(val_loader))

with torch.no_grad():
    prediction = model(sample_X)
    original_pred = target_scaler.inverse_transform(prediction.numpy())
    original_true = target_scaler.inverse_transform(sample_y.numpy())

    print(f"\nSample Prediction:")
    print(f"Predicted value: {original_pred[0][0]:.4f}")
    print(f"True value: {original_true[0][0]:.4f}")