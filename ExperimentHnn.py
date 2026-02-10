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
import matplotlib.pyplot as plt
from Config import paths  # Assuming Config.py contains paths configuration

# =============================================
# CONFIGURATION AND UTILITY FUNCTIONS
# =============================================

# Get main folder path from configuration
main_folder = paths['mainfolder']


def parse_value(x):
    """
    Parse a value from string representation to Python object.
    Handles lists formatted as strings (e.g., "[1, 2, 3]").

    Args:
        x: Value to parse (string or other type)

    Returns:
        Parsed Python object or original value if parsing fails
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
        List of lists with proper float32 values
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

    Directory structure expected:
    main_folder/
        videos01/
            mask1/
            mask2/
        videos02/
            mask1/

    Args:
        directory_path: Root directory containing video folders

    Returns:
        List of paths to mask subdirectories
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"'{directory_path}' is not a valid directory")

    # Patterns to match video directories (videos01, videos02, etc.)
    video_pattern = re.compile(r'^videos([1-2]?[0-9])$')
    # Pattern to match mask directories (mask1, mask2, etc.)
    mask_pattern = re.compile(r'^mask\d+$')

    mask_subdirs = []

    # Iterate through all items in the main directory
    for video_dir in os.listdir(directory_path):
        video_path = os.path.join(directory_path, video_dir)
        # Check if it's a video directory matching the pattern
        if os.path.isdir(video_path) and video_pattern.match(video_dir):
            for subdir in os.listdir(video_path):
                subdir_path = os.path.join(video_path, subdir)
                # Check if it's a mask directory matching the pattern
                if os.path.isdir(subdir_path) and mask_pattern.match(subdir):
                    mask_subdirs.append(subdir_path)

    return sorted(mask_subdirs)


def getData():
    """
    Main function to load and preprocess data from all mask directories.

    Returns:
        Tuple of (X, y) where:
            X: Feature matrix of shape (n_samples, 76)
            y: Target values of shape (n_samples,)
    """
    xTrain_combined = []  # List to store feature arrays from all masks
    yTrain_combined = []  # List to store target arrays from all masks

    # Process each mask directory
    for mask in get_mask_subdirs_os2(main_folder):
        print("##################################################")
        print(f"Processing mask directory: {mask}")

        # Load the CSV file containing training data
        df = pd.read_csv(mask + '/trainDataHnn3step.csv')

        # ======================
        # Process 'cap' column data
        # ======================
        capData = []
        for row in df['cap']:
            print("Processing cap data...")

            # Clean up the string by replacing np.float32 with float
            row = row.replace("np.float32", "float")
            # Remove float() wrappers
            row = re.sub(r"float\((-?\d*\.?\d+(?:e[-+]?\d+)?)\)", r"\1", row)

            # Define patterns to clean various array representations
            patterns = [
                (r"array\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*dtype=float32\)", r"\1"),
                (r"array\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*dtype=float64\)", r"\1"),
                (r"np\.float32\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)", r"\1"),
                (r"np\.float64\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)", r"\1"),
                (r"array\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)", r"\1")
            ]

            print("Before regex cleanup:", row)

            # Apply all cleaning patterns
            for pattern, replacement in patterns:
                row = re.sub(pattern, replacement, row, flags=re.IGNORECASE)

            print("After regex cleanup:", row)

            try:
                # Parse the cleaned string into a Python list
                print("@ Parsing row:", row)
                parsed_data = eval(row)  # Note: Using eval can be unsafe with untrusted data
                # Convert to list of lists with float32 values
                result = [[item[0],
                           np.float32(item[1]),
                           np.float32(item[2]),
                           np.float32(item[3]),
                           np.float32(item[4])]
                          for item in parsed_data]
                print("Parsed result:", result)

            except Exception as e:
                print(f"Error parsing string: {e}")
                print(f"Problematic string: {row}")
                print(f"Problematic mask: {mask}")
                raise

            # Process each point to create fixed-length representation
            cap = []
            index = 1
            for point in result:
                while True:
                    if point[0] == index:
                        # Add the 4 feature values for this point
                        cap.append(point[1:])
                        index += 1
                        break
                    elif point[0] != index:
                        # Fill missing points with zeros
                        cap.append([0, 0, 0, 0])
                        index += 1
                        if index == 10:  # Maximum of 9 points expected
                            break

            # Fill remaining slots with zeros if we have fewer than 9 points
            while index < 10:
                cap.append([0, 0, 0, 0])
                index += 1

            # Flatten the 2D list to 1D (9 points × 4 features = 36 values)
            flat_list = [item for sublist in cap for item in sublist]
            print("Final flattened cap data:", flat_list, "Length:", len(flat_list))
            capData.append(flat_list)

        # ======================
        # Process 'hnnvoordinates' column data (similar to 'cap')
        # ======================
        capHnnData = []
        for row in df['hnnvoordinates']:
            print("Processing hnnvoordinates data...")

            # Similar cleaning process as above
            row = row.replace("np.float32", "float")
            row = re.sub(r"float\((-?\d*\.?\d+(?:e[-+]?\d+)?)\)", r"\1", row)

            # Extended patterns for various array representations
            patterns = [
                (r"array\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*dtype=float32\)", r"\1"),
                (r"array\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*dtype=float64\)", r"\1"),
                (r"np\.float32\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)", r"\1"),
                (r"np\.float64\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)", r"\1"),
                (r"array\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)", r"\1"),
                # Additional patterns for edge cases
                (r"array\(([-+]?\d+\.)\s*,\s*dtype=float32\)", r"\1"),
                (r"array\(([-+]?\d+\.)\s*,\s*dtype=float64\)", r"\1"),
                (r"np\.float32\(([-+]?\d+\.)\s*\)", r"\1"),
                (r"np\.float64\(([-+]?\d+\.)\s*\)", r"\1"),
                (r"array\(([-+]?\d+\.)\s*\)", r"\1")
            ]

            print("Before regex cleanup:", row)

            for pattern, replacement in patterns:
                row = re.sub(pattern, replacement, row, flags=re.IGNORECASE)
            print("After regex cleanup:", row)

            try:
                # Parse using ast.literal_eval for safety
                parsed_data = ast.literal_eval(row)
                print("Parsed data:", parsed_data)

                # Convert to list of lists with float32 values
                result = [[np.float32(item[0]),
                           np.float32(item[1]),
                           np.float32(item[2]),
                           np.float32(item[3]),
                           np.float32(item[4])]
                          for item in parsed_data]

            except ValueError as e:
                print(f"Error parsing string: {e}")
                print(f"Problematic string: {row}")
                raise

            # Same fixed-length processing as before
            cap = []
            index = 1
            for point in result:
                print("Processing point:", point)
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

            # Flatten to 36 values (9×4)
            flat_list = [item for sublist in cap for item in sublist]
            print("Final flattened hnn data:", flat_list, "Length:", len(flat_list))
            capHnnData.append(flat_list)

        # ======================
        # Process 'coordinates' column (bounding box data)
        # ======================
        centerSizeData = []
        for row in df['coordinates']:
            # Parse bounding box coordinates [x, y, width, height]
            parsed_data = ast.literal_eval(row)
            # Calculate size and center coordinates
            size = parsed_data[2] * parsed_data[3]  # width × height
            centerX = parsed_data[0] + (parsed_data[2] / 2)  # x + width/2
            centerY = parsed_data[1] + (parsed_data[3] / 2)  # y + height/2
            print(f"Bounding box data - Size: {size}, CenterX: {centerX}, CenterY: {centerY}")
            centerSizeData.append([np.float32(size), np.float32(centerX), np.float32(centerY)])

        # ======================
        # Process 'frameNum' column
        # ======================
        frameNumData = []
        for row in df['frameNum']:
            # Adjust frame number slightly (to avoid exact integers?)
            adjusted_frame = np.float32(row - 0.0001)
            print(f"Frame number: {row} -> {adjusted_frame}")
            frameNumData.append([adjusted_frame])

        # ======================
        # Process 'score' column (target variable)
        # ======================
        scoreData = []
        for row in df['score']:
            print(f"Score: {row}")
            scoreData.append(row)

        # ======================
        # Combine all features
        # ======================
        combined = []
        for l1, l2, l3, l4 in zip(capData, centerSizeData, frameNumData, capHnnData):
            print(f"Combining: capData({len(l1)}), bbox({len(l2)}), frame({len(l3)}), hnn({len(l4)})")
            # Total features: 36 (cap) + 3 (bbox) + 1 (frame) + 36 (hnn) = 76
            combined.append(l1 + l2 + l3 + l4)

        print(f"First combined sample: {combined[0]}, Length: {len(combined[0])}")

        # Convert to numpy arrays
        X_array = np.array(combined)
        y_array = np.array(scoreData)

        print(f"X_array shape: {X_array.shape}, y_array shape: {y_array.shape}")

        # Add to combined lists
        xTrain_combined.append(X_array)
        yTrain_combined.append(y_array)

    # ======================
    # Combine data from all masks
    # ======================
    xTrain_final = np.concatenate(xTrain_combined, axis=0)
    yTrain_final = np.concatenate(yTrain_combined, axis=0)

    # Verify final shapes
    print(f"Final X shape: {xTrain_final.shape}")
    print(f"Final y shape: {yTrain_final.shape}")

    return xTrain_final, yTrain_final


# =============================================
# MAIN EXECUTION
# =============================================

# Load the data
X, y = getData()
print(f"Loaded data - X shape: {X.shape}, y shape: {y.shape}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

# Standardize features (zero mean, unit variance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================
# BUILD NEURAL NETWORK
# ======================

model = Sequential([
    # Input layer: 76 features
    Dense(256, activation='relu', input_shape=(76,)),
    BatchNormalization(),  # Normalize activations
    Dropout(0.3),  # Regularization: randomly drop 30% of neurons

    # Hidden layers
    Dense(228, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(228, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(228, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    # Additional hidden layer
    Dense(16, activation='relu'),

    # Output layer: single value with sigmoid activation (0-1 range)
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mean_squared_error',  # MSE for regression
    metrics=['mae']  # Mean Absolute Error as additional metric
)

# Display model architecture
model.summary()

# ======================
# TRAIN THE MODEL
# ======================

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=75,
    batch_size=32,
    verbose=1
)

# ======================
# EVALUATE THE MODEL
# ======================

# Evaluate on test set
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss (MSE): {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# Make predictions on a few samples
predictions = model.predict(X_test[:5])
print(f"\nSample predictions on first 5 test samples:")
print(predictions.flatten())

# ======================
# SAVE THE MODEL
# ======================

# Save model for later use
model.save('nn_model.h5')
print("Model saved as 'nn_model.h5'")

# ======================
# VISUALIZE TRAINING
# ======================

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', color='#1f77b4', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', color='#ff7f0e', linewidth=2)
plt.title('Training and Validation Loss Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Mean Squared Error Loss', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()