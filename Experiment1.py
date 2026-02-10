# Import necessary libraries
import os
import re
import numpy as np
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
from Config import paths

# Get main folder path from configuration file
main_folder = paths['mainfolder']


def parse_value(x):
    """
    Parse a value, converting string representations of lists to actual lists.

    Args:
        x: Input value (could be string, list, or other type)

    Returns:
        Parsed value - if string represents a list, returns the list; otherwise returns original value
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
    Parse custom array strings with np.float32() wrappers.

    Args:
        s: String representation of array with np.float32() wrappers

    Returns:
        List of lists with proper float32 conversion
    """
    # Remove np.float32() wrappers while keeping the numeric values
    cleaned = re.sub(r'np\.float32\(([^)]+)\)', r'\1', s)
    # Convert to actual list
    data = ast.literal_eval(cleaned)
    # Convert numbers to float32 if needed
    return [[x[0]] + [np.float32(y) for y in x[1:]] for x in data]


def get_mask_subdirs_os2(directory_path):
    """
    Find all mask subdirectories within video directories.

    Args:
        directory_path: Path to the main directory containing video folders

    Returns:
        Sorted list of paths to mask subdirectories
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"'{directory_path}' is not a valid directory")

    # Regex patterns for matching directory names
    video_pattern = re.compile(r'^videos([1-2]?[0-9])$')
    mask_pattern = re.compile(r'^mask\d+$')

    mask_subdirs = []

    # Iterate through video directories
    for video_dir in os.listdir(directory_path):
        video_path = os.path.join(directory_path, video_dir)
        if os.path.isdir(video_path) and video_pattern.match(video_dir):
            # Look for mask subdirectories within each video directory
            for subdir in os.listdir(video_path):
                subdir_path = os.path.join(video_path, subdir)
                if os.path.isdir(subdir_path) and mask_pattern.match(subdir):
                    mask_subdirs.append(subdir_path)

    return sorted(mask_subdirs)


def getData():
    """
    Load and preprocess training data from multiple mask directories.

    Returns:
        X_train_final: Combined feature matrix from all mask directories
        y_train_final: Combined target values from all mask directories
    """
    xTrain_combined = []
    yTrain_combined = []

    # Iterate through all mask directories
    for mask in get_mask_subdirs_os2(main_folder):
        print(f"Processing mask directory: {mask}")

        # Load CSV data for current mask
        df = pd.read_csv(mask + '/trainData.csv')

        # Process cap (capacity) data
        capData = []
        for row in df['cap']:
            # Clean the string: remove float wrappers and convert to proper format
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

            # Process each point in the cap data
            cap = []
            index = 1
            for point in result:
                while True:
                    if point[0] == index:
                        # Add the point data
                        cap.append(point[1:])
                        index += 1
                        break
                    elif point[0] != index:
                        # Add zero padding for missing points
                        cap.append([0, 0, 0, 0])
                        index += 1
                        if index == 10:  # Limit to 9 points
                            break

            # Pad with zeros if we have less than 9 points
            while index < 10:
                cap.append([0, 0, 0, 0])
                index += 1

            # Flatten the cap data to 1D list (9 points * 4 values = 36 features)
            flat_list = [item for sublist in cap for item in sublist]
            capData.append(flat_list)

        # Process coordinates data (center and size calculations)
        centerSizeData = []
        for row in df['coordinates']:
            # Clean the string
            row = row.replace("np.float32", "float")
            row = re.sub(r"float\((-?\d*\.?\d+(?:e[-+]?\d+)?)\)", r"\1", row)

            try:
                # Parse coordinates string
                parsed_data = ast.literal_eval(row)
            except ValueError as e:
                print(f"Error parsing string: {e}")
                print(f"Problematic string: {row}")
                raise

            # Calculate size (area) and center coordinates
            size = parsed_data[2] * parsed_data[3]
            centerX = parsed_data[0] + (parsed_data[2] / 2)
            centerY = parsed_data[1] + (parsed_data[3] / 2)
            centerSizeData.append([np.float32(size), np.float32(centerX), np.float32(centerY)])

        # Process frame number data
        frameNumData = []
        for row in df['frameNum']:
            frameNumData.append([np.float32(row - 0.0001)])  # Small offset for numerical stability

        # Process target score data
        scoreData = []
        for row in df['score']:
            scoreData.append(row)

        # Combine all features into single feature vectors
        combined = []
        for l1, l2, l3 in zip(capData, centerSizeData, frameNumData):
            # Combine cap features (36) + center/size features (3) + frame number (1) = 40 total features
            combined.append(l1 + l2 + l3)

        # Convert to numpy arrays
        X_array = np.array(combined)
        y_array = np.array(scoreData)

        # Add to combined lists
        xTrain_combined.append(X_array)
        yTrain_combined.append(y_array)

    # Concatenate data from all mask directories
    xTrain_final = np.concatenate(xTrain_combined, axis=0)
    yTrain_final = np.concatenate(yTrain_combined, axis=0)

    # Print final shapes for verification
    print("xTrain_final shape:", xTrain_final.shape)
    print("yTrain_final shape:", yTrain_final.shape)

    return xTrain_final, yTrain_final


# Main execution
# Load data
X, y = getData()
print(f"Loaded data shape - X: {X.shape}, y: {y.shape}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training data shape: {X_train.shape}")

# Scale the features for neural network training
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network model
model = Sequential([
    Dense(256, activation='relu', input_shape=(40,)),  # Input layer: 40 features
    BatchNormalization(),  # Normalize activations for faster training
    Dense(228, activation='relu'),  # Hidden layer 1
    BatchNormalization(),
    Dense(228, activation='relu'),  # Hidden layer 2
    BatchNormalization(),
    Dense(228, activation='relu'),  # Hidden layer 3
    BatchNormalization(),
    Dense(16, activation='relu'),  # Hidden layer 4 (bottleneck)
    Dense(1, activation='sigmoid')  # Output layer: single value between 0-1
])

# Compile the model with Adam optimizer and MSE loss
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mean_squared_error',
              metrics=['mae'])  # Mean Absolute Error for evaluation

# Display model architecture
model.summary()

# Train the model
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=75,
                    batch_size=32,
                    verbose=1)

# Evaluate the model on test data
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest MAE: {test_mae:.4f}")

# Make sample predictions
predictions = model.predict(X_test[:5])
print("\nSample predictions:", predictions.flatten())

# Save the trained model
model.save('nn_model.h5')
print("Model saved as 'nn_model.h5'")

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', color='#1f77b4')
plt.plot(history.history['val_loss'], label='Validation Loss', color='#ff7f0e')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error Loss')
plt.legend()
plt.grid(True)
plt.show()