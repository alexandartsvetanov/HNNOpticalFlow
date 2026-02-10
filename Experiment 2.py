# Import necessary libraries
import os
import re
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Machine learning and deep learning imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Import configuration paths
from Config import paths

# Get main folder path from configuration
main_folder = paths['mainfolder']


def create_dataset(data, window_size=10):
    """
    Create sequences for time-series data using sliding window approach.

    Parameters:
    data (np.array): Input data of shape (n_samples, n_features)
    window_size (int): Number of consecutive samples per sequence

    Returns:
    np.array: 3D array of shape (n_sequences, window_size, n_features)
    """
    sequences = []
    print(f"Input data shape: {data.shape}")

    # Create sequences using a sliding window
    for i in range(data.shape[0] - window_size):
        # Extract a sequence of 'window_size' rows starting at index i
        sequence = data[i:i + window_size]  # Shape: (window_size, n_features)
        print(f"Sequence shape: {sequence.shape}")
        sequences.append(sequence)

    # Convert list of sequences to a 3D numpy array
    return np.array(sequences)


def parse_value(x):
    """
    Parse string values, converting string representations of lists to actual lists.

    Parameters:
    x: Input value (could be string, list, or other type)

    Returns:
    Parsed value
    """
    if isinstance(x, str):
        try:
            # Try to parse string as a list (e.g., "[1, 2, 3]" -> [1, 2, 3])
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            # If it's not a valid list, treat it as a single value
            return x
    return x


def parse_custom_array(s):
    """
    Parse custom array strings, removing np.float32() wrappers.

    Parameters:
    s (str): String representation of array

    Returns:
    list: Parsed list with float32 values
    """
    # Remove np.float32() wrappers while keeping the numeric values
    cleaned = re.sub(r'np\.float32\(([^)]+)\)', r'\1', s)
    # Convert to actual list
    data = ast.literal_eval(cleaned)
    # Convert numbers to float32 if needed
    return [[x[0]] + [np.float32(y) for y in x[1:]] for x in data]


def get_mask_subdirs_os2(directory_path):
    """
    Recursively find all mask subdirectories within the specified directory structure.

    Directory structure expected:
    main_folder/
        videos01/ (or videos1, videos2, etc.)
            mask1/
            mask2/
        videos02/
            mask1/

    Parameters:
    directory_path (str): Path to the main directory

    Returns:
    list: Sorted list of paths to mask subdirectories
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"'{directory_path}' is not a valid directory")

    # Regex patterns to match directory names
    video_pattern = re.compile(r'^videos([1-2]?[0-9])$')
    mask_pattern = re.compile(r'^mask\d+$')

    mask_subdirs = []

    # Iterate through video directories
    for video_dir in os.listdir(directory_path):
        video_path = os.path.join(directory_path, video_dir)

        # Check if directory matches video pattern
        if os.path.isdir(video_path) and video_pattern.match(video_dir):
            # Iterate through subdirectories in video directory
            for subdir in os.listdir(video_path):
                subdir_path = os.path.join(video_path, subdir)

                # Check if subdirectory matches mask pattern
                if os.path.isdir(subdir_path) and mask_pattern.match(subdir):
                    mask_subdirs.append(subdir_path)

    return sorted(mask_subdirs)


def getData():
    """
    Main function to load and preprocess data from all mask directories.

    Returns:
    tuple: (xTrain_final, yTrain_final) - Combined and processed training data
    """
    xTrain_combined = []
    yTrain_combined = []

    # Process each mask directory
    for mask in get_mask_subdirs_os2(main_folder):
        print(f"Processing directory: {mask}")

        # Load training data CSV
        df = pd.read_csv(mask + '/trainData.csv')

        # Process 'cap' column - contains capacity/feature data
        capData = []
        for row in df['cap']:
            # Clean up the string representation
            row = row.replace("np.float32", "float")
            row = re.sub(r"float\((-?\d*\.?\d+(?:e[-+]?\d+)?)\)", r"\1", row)

            try:
                # Parse the string into a Python list
                parsed_data = ast.literal_eval(row)

                # Convert float values to np.float32
                result = [[item[0], np.float32(item[1]), np.float32(item[2]),
                           np.float32(item[3]), np.float32(item[4])]
                          for item in parsed_data]

            except ValueError as e:
                print(f"Error parsing string: {e}")
                print(f"Problematic string: {row}")
                raise

            # Reformat data: fill missing indices with zeros
            cap = []
            index = 1
            for point in result:
                while True:
                    if point[0] == index:
                        cap.append(point[1:])  # Append features (4 values)
                        index += 1
                        break
                    elif point[0] != index:
                        cap.append([0, 0, 0, 0])  # Fill missing with zeros
                        index += 1
                        if index == 10:
                            break

            # Fill any remaining positions with zeros
            while index < 10:
                cap.append([0, 0, 0, 0])
                index += 1

            # Flatten the 2D list to 1D
            flat_list = [item for sublist in cap for item in sublist]
            capData.append(flat_list)

        # Process 'coordinates' column - extract size and center coordinates
        centerSizeData = []
        for row in df['coordinates']:
            # Clean up the string
            row = row.replace("np.float32", "float")
            row = re.sub(r"float\((-?\d*\.?\d+(?:e[-+]?\d+)?)\)", r"\1", row)

            try:
                # Parse the string into a Python list
                parsed_data = ast.literal_eval(row)
            except ValueError as e:
                print(f"Error parsing string: {e}")
                print(f"Problematic string: {row}")
                raise

            # Calculate size and center coordinates
            size = parsed_data[2] * parsed_data[3]  # width * height
            centerX = parsed_data[0] + (parsed_data[2] / 2)  # x + width/2
            centerY = parsed_data[1] + (parsed_data[3] / 2)  # y + height/2

            centerSizeData.append([np.float32(size), np.float32(centerX), np.float32(centerY)])

        # Process 'frameNum' column
        frameNumData = []
        for row in df['frameNum']:
            # Add small offset to frame numbers
            frameNumData.append([np.float32(row - 0.0001)])

        # Process 'score' column - target values
        scoreData = []
        for row in df['score']:
            scoreData.append(row)

        # Combine all features into single arrays
        combined = []
        for l1, l2, l3 in zip(capData, centerSizeData, frameNumData):
            combined.append(l1 + l2 + l3)

        print(f"First combined sample: {combined[0]}, Length: {len(combined[0])}")

        # Convert to numpy array
        X_array = np.array(combined)

        # Skip sequences that are too short for the window
        if X_array.shape[0] < 10:
            continue

        print(f"X_array shape before creating sequences: {X_array.shape}")

        # Create time-series sequences
        X_array = create_dataset(X_array)

        # Align target values with sequences (skip first window_size samples)
        y_array = np.array(scoreData[10:])

        print(f"X_array shape after creating sequences: {X_array.shape}")
        print(f"y_array shape: {y_array.shape}")

        # Store data from this mask directory
        xTrain_combined.append(X_array)
        yTrain_combined.append(y_array)

    # Combine data from all mask directories
    xTrain_final = np.concatenate(xTrain_combined, axis=0)
    yTrain_final = np.concatenate(yTrain_combined, axis=0)

    # Verify final shapes
    print(f"Final combined X shape: {xTrain_final.shape}")
    print(f"Final combined y shape: {yTrain_final.shape}")

    return xTrain_final, yTrain_final


# Main execution block
if __name__ == "__main__":
    # Load and preprocess data
    X, y = getData()

    # Reshape y to 2D array (n_samples, 1)
    y = y.reshape(-1, 1)

    # Reshape X to 2D for scaling, then back to 3D
    X_reshaped = X.reshape(-1, X.shape[-1])

    # Initialize and fit scaler
    scaler = StandardScaler()
    X_scaled_2d = scaler.fit_transform(X_reshaped)

    # Reshape back to 3D: (n_sequences, window_size, n_features)
    X_scaled = X_scaled_2d.reshape(X.shape)

    print(f"Final X shape: {X.shape}")
    print(f"Final y shape: {y.shape}")

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    print(f"Train shapes - X: {X_train.shape}, y: {y_train.shape}")
    print(f"Validation shapes - X: {X_val.shape}, y: {y_val.shape}")

    # Define LSTM model architecture
    model = Sequential([
        # LSTM layer with 64 units, input shape matches (sequence_length, features)
        LSTM(64, input_shape=(10, 40), return_sequences=False),

        # Dropout for regularization
        Dropout(0.2),

        # Dense layer with ReLU activation
        Dense(32, activation='relu'),

        # Output layer with linear activation (for regression)
        Dense(y.shape[1], activation='linear')
    ])

    # Compile the model with MSE loss and MAE metric
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Define early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=4,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate the model on validation data
    val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}")

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()