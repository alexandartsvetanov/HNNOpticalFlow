import os
import re
import numpy as np
import pandas as pd
import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
#from pythonProject7.podgotovkaPandas import X_train
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def create_dataset(data, window_size=10):
    sequences = []
    print(data.shape)
    # Create sequences using a sliding window
    for i in range(data.shape[0]-window_size):
        # Extract a sequence of 'sequence_length' rows starting at index i
        sequence = data[i:i + window_size]  # Shape: (10, 40)
        print(sequence.shape)
        sequences.append(sequence)

    # Convert list of sequences to a 3D numpy array
    return np.array(sequences)

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
            #print(flat_list, len(flat_list))
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
            #print(size, centerX, centerY)
            centerSizeData.append([np.float32(size), np.float32(centerX), np.float32(centerY)])

        frameNumData = []
        for row in df['frameNum']:
            #print(np.float32(row - 0.0001))
            frameNumData.append([np.float32(row - 0.0001)])

        scoreData = []
        for row in df['score']:
            #print(row)
            scoreData.append(row)

        combined = []
        for l1, l2, l3 in zip(capData, centerSizeData, frameNumData):
            combined.append(l1 + l2 + l3)
        print(combined[0], len(combined[0]))
        X_array = np.array(combined)
        if X_array.shape[0] < 10:
            continue
        print(X_array.shape)
        X_array = create_dataset(X_array)
        y_array = np.array(scoreData[10:])
        print(X_array.shape)

        print(y_array.shape)

        #dataEncoded = encode_pairs(dataset)
        xTrain_combined.append(X_array)
        yTrain_combined.append(y_array)

    xTrain_final = np.concatenate(xTrain_combined, axis=0)  # or np.vstack(xTrain_combined)
    yTrain_final = np.concatenate(yTrain_combined, axis=0)  # or np.vstack(yTrain_combined)

        # Verify shapes
    print("xTrain_final shape:", xTrain_final.shape)
    print("yTrain_final shape:", yTrain_final.shape)



    return xTrain_final, yTrain_final

X, y = getData()
y = y.reshape(-1, 1)
X_reshaped = X.reshape(-1, X.shape[-1])

# Fit and transform the data
scaler = StandardScaler()
X_scaled_2d = scaler.fit_transform(X_reshaped)

# Reshape back to 3D: (20, 10, 40)
X_scaled = X_scaled_2d.reshape(X.shape)
print(X.shape)
print(y.shape)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

print("Train shapes:", X_train.shape, y_train.shape)  # e.g., (16, 10, 40), (16, n_outputs)
print("Validation shapes:", X_val.shape, y_val.shape)




# Define the model
model = Sequential([
    LSTM(64, input_shape=(10, 40), return_sequences=False),  # 64 units
    Dropout(0.2),  # Prevent overfitting
    Dense(32, activation='relu'),
    Dense(y.shape[1], activation='linear')  # Matches number of outputs
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=4,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}")




plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()