"""
Sequence Prediction Model for Time-Series Data
This script processes sequential data from multiple video masks and trains an RNN model for regression.
"""

import os
import re
import numpy as np
import pandas as pd
import ast
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
# Note: Some imports are duplicated in the original code but kept for compatibility

# Configuration import (assumes Config.py exists with 'paths' dictionary)
from Config import paths

# =============================================
# CONFIGURATION AND UTILITY FUNCTIONS
# =============================================

# Get main folder path from configuration
main_folder = paths['mainfolder']


def parse_value(x):
    """
    Parse a string value that might contain Python literals (like lists).

    Args:
        x: Input value (could be string or other type)

    Returns:
        Parsed Python object if string contains valid literal, otherwise original value
    """
    if isinstance(x, str):
        try:
            # Try to parse string as a Python literal (e.g., "[1, 2, 3]")
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            # If parsing fails, return the string as-is
            return x
    return x


def parse_custom_array(s):
    """
    Parse custom string format containing np.float32() wrappers.

    Args:
        s: Input string with np.float32() patterns

    Returns:
        List with np.float32 values
    """
    # Remove np.float32() wrappers while keeping numeric values
    cleaned = re.sub(r'np\.float32\(([^)]+)\)', r'\1', s)
    # Convert string to actual Python list
    data = ast.literal_eval(cleaned)
    # Convert numbers to float32
    return [[x[0]] + [np.float32(y) for y in x[1:]] for x in data]


def get_mask_subdirs_os2(directory_path):
    """
    Recursively find all mask subdirectories within video directories.

    Expected structure:
    directory_path/
    ├── videos01/
    │   ├── mask1/
    │   └── mask2/
    ├── videos02/
    │   ├── mask1/
    │   └── mask3/

    Args:
        directory_path: Root directory containing video folders

    Returns:
        List of paths to mask subdirectories
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"'{directory_path}' is not a valid directory")

    # Regex patterns for matching directory names
    video_pattern = re.compile(r'^videos([1-2]?[0-9])$')  # Matches videos01, videos02, etc.
    mask_pattern = re.compile(r'^mask\d+$')  # Matches mask1, mask2, etc.

    mask_subdirs = []

    # Iterate through video directories
    for video_dir in os.listdir(directory_path):
        video_path = os.path.join(directory_path, video_dir)
        if os.path.isdir(video_path) and video_pattern.match(video_dir):
            # Look for mask subdirectories within video directory
            for subdir in os.listdir(video_path):
                subdir_path = os.path.join(video_path, subdir)
                if os.path.isdir(subdir_path) and mask_pattern.match(subdir):
                    mask_subdirs.append(subdir_path)

    return sorted(mask_subdirs)  # Return sorted for consistent ordering


def getData():
    """
    Main data loading function that processes all mask directories.

    Returns:
        xTrain_combined: List of numpy arrays (each shape: [timesteps, 43 features])
        yTrain_combined: List of numpy arrays (each shape: [timesteps,])
    """
    xTrain_combined = []
    yTrain_combined = []

    # Process each mask directory
    for mask in get_mask_subdirs_os2(main_folder):
        print(f"Processing: {mask}")

        # Load the training data CSV
        df = pd.read_csv(mask + '/trainData.csv')

        # ==================== Process 'cap' column ====================
        capData = []
        for row in df['cap']:
            # Clean up the string representation
            row = row.replace("np.float32", "float")
            row = re.sub(r"float\((-?\d*\.?\d+(?:e[-+]?\d+)?)\)", r"\1", row)

            try:
                # Parse string to Python list
                parsed_data = ast.literal_eval(row)

                # Convert to np.float32 and restructure
                result = [[item[0], np.float32(item[1]), np.float32(item[2]),
                           np.float32(item[3]), np.float32(item[4])]
                          for item in parsed_data]

            except ValueError as e:
                print(f"Error parsing string: {e}")
                print(f"Problematic string: {row}")
                raise

            # Process each point to ensure fixed length (9 points, each with 4 features)
            cap = []
            index = 1
            for point in result:
                while True:
                    if point[0] == index:
                        # Add actual point data
                        cap.append(point[1:])  # Get features (skip index)
                        index += 1
                        break
                    elif point[0] != index:
                        # Add zero padding for missing points
                        cap.append([0, 0, 0, 0])
                        index += 1
                        if index == 10:  # Stop after 9 points
                            break

            # Pad with zeros if we have fewer than 9 points
            while index < 10:
                cap.append([0, 0, 0, 0])
                index += 1

            # Flatten the 9x4 matrix to 36-element vector
            flat_list = [item for sublist in cap for item in sublist]
            capData.append(flat_list)

        # ==================== Process 'coordinates' column ====================
        centerSizeData = []
        for row in df['coordinates']:
            # Similar cleaning as above
            row = row.replace("np.float32", "float")
            row = re.sub(r"float\((-?\d*\.?\d+(?:e[-+]?\d+)?)\)", r"\1", row)

            try:
                parsed_data = ast.literal_eval(row)
            except ValueError as e:
                print(f"Error parsing string: {e}")
                print(f"Problematic string: {row}")
                raise

            # Calculate derived features from bounding box
            size = parsed_data[2] * parsed_data[3]  # width * height
            centerX = parsed_data[0] + (parsed_data[2] / 2)  # x + width/2
            centerY = parsed_data[1] + (parsed_data[3] / 2)  # y + height/2

            centerSizeData.append([np.float32(size), np.float32(centerX), np.float32(centerY)])

        # ==================== Process 'score' column ====================
        scoreData = []
        for row in df['score']:
            scoreData.append(row)

        # ==================== Combine features ====================
        # Combine cap features (36) with center/size features (3) = 39 total features
        combined = []
        for l1, l2 in zip(capData, centerSizeData):
            combined.append(l1 + l2)  # 36 + 3 = 39 features

        # Convert to numpy arrays
        X_array = np.array(combined)  # Shape: (n_timesteps, 39)
        y_array = np.array(scoreData)  # Shape: (n_timesteps,)

        # Store this mask's data
        xTrain_combined.append(X_array)
        yTrain_combined.append(y_array)

    return xTrain_combined, yTrain_combined


# =============================================
# DATA LOADING
# =============================================

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load all data from mask directories
X, y = getData()
print(f"Loaded {len(X)} sequences")
print(f"First sequence shape: {X[0].shape}")


# =============================================
# DATASET CLASS DEFINITION
# =============================================

class SequenceDataset(Dataset):
    """
    Custom PyTorch Dataset for sequence prediction tasks.
    Creates sliding windows from time-series data.
    """

    def __init__(self, sequences, targets, window_size=10):
        """
        Initialize dataset with sequences and targets.

        Args:
            sequences: List of numpy arrays (each shape: [timesteps, features])
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

            # Create sliding windows
            for i in range(seq_length - window_size + 1):
                window = seq[i:i + window_size]  # Shape: (window_size, features)
                target = target_seq[i + window_size - 1]  # Predict value at last timestep

                self.X.append(window)
                self.y.append(target)

        # Convert to numpy arrays
        self.X = np.stack(self.X)  # Shape: (n_samples, window_size, features)
        self.y = np.array(self.y).reshape(-1, 1)  # Shape: (n_samples, 1)

        # Normalize features (per-feature standardization)
        self.scaler_X = StandardScaler()
        original_shape = self.X.shape
        # Reshape to 2D for scaling, then reshape back to 3D
        self.X = self.scaler_X.fit_transform(
            self.X.reshape(-1, self.features)).reshape(original_shape)

        # Normalize targets
        self.scaler_y = StandardScaler()
        self.y = self.scaler_y.fit_transform(self.y)

    def __len__(self):
        """Return total number of samples in dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Returns:
            x: Input tensor of shape (window_size, features)
            y: Target tensor of shape (1,)
        """
        x = torch.FloatTensor(self.X[idx])
        y = torch.FloatTensor(self.y[idx])
        return x, y


# =============================================
# MODEL DEFINITION
# =============================================

class RNNModel(nn.Module):
    """
    RNN model using GRU layers for sequence regression.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Initialize RNN model.

        Args:
            input_size: Number of input features per timestep
            hidden_size: Size of hidden state in GRU
            num_layers: Number of stacked GRU layers
            output_size: Number of output values (1 for regression)
        """
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU layer for sequence processing
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, window_size, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden state with zeros
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # Forward pass through GRU
        out, _ = self.gru(x, h0)  # out shape: (batch_size, window_size, hidden_size)

        # Use output from the last timestep only
        out = out[:, -1, :]  # Shape: (batch_size, hidden_size)

        # Fully connected layer for final prediction
        out = self.fc(out)  # Shape: (batch_size, output_size)
        return out


# =============================================
# TRAINING FUNCTION
# =============================================

def train_model(dataset, epochs=100, batch_size=32, learning_rate=0.001):
    """
    Train the RNN model on the provided dataset.

    Args:
        dataset: SequenceDataset object
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer

    Returns:
        model: Trained PyTorch model
        target_scaler: Scaler object for inverse transforming predictions
    """
    # Split dataset into training and validation sets (80/20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    input_size = dataset.features
    hidden_size = 64
    num_layers = 2
    output_size = 1

    model = RNNModel(input_size, hidden_size, num_layers, output_size)

    # Loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Track losses for plotting
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        epoch_train_loss = 0

        # Training phase
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        # Validation phase
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                epoch_val_loss += loss.item()

        # Calculate average losses
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}')

    # Plot training history
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


# =============================================
# MAIN EXECUTION
# =============================================

if __name__ == "__main__":
    # 1. Create dataset with sliding windows
    window_size = 10
    dataset = SequenceDataset(X, y, window_size=window_size)
    print(f"Dataset created with {len(dataset)} samples")
    print(f"Input shape: {dataset.X.shape}")  # (n_samples, window_size, features)

    # 2. Train the model
    print("\nStarting training...")
    model, target_scaler = train_model(
        dataset,
        epochs=50,
        batch_size=32,
        learning_rate=0.001
    )

    # 3. Make a sample prediction
    print("\nMaking sample prediction...")
    val_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    sample_X, sample_y = next(iter(val_loader))

    with torch.no_grad():
        prediction = model(sample_X)
        # Inverse transform to get original scale
        original_pred = target_scaler.inverse_transform(prediction.numpy())
        original_true = target_scaler.inverse_transform(sample_y.numpy())

        print(f"\nSample Prediction:")
        print(f"Predicted value: {original_pred[0][0]:.4f}")
        print(f"True value: {original_true[0][0]:.4f}")
        print(f"Error: {abs(original_pred[0][0] - original_true[0][0]):.4f}")