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
        print("##################################################")
        print(mask)
        df = pd.read_csv(mask + '/trainDataHnn3step.csv')
        capData = []
        for row in df['cap']:
            print("nachalo")
            # Single pattern that handles ALL cases with global replacement
            row = row.replace("np.float32", "float")
            row = re.sub(r"float\((-?\d*\.?\d+(?:e[-+]?\d+)?)\)", r"\1", row)
            print("nachalo2")
            patterns = [
                (r"array\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*dtype=float32\)", r"\1"),
                (r"array\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*dtype=float64\)", r"\1"),
                (r"np\.float32\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)", r"\1"),
                (r"np\.float64\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)", r"\1"),
                (r"array\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)", r"\1")
            ]

            print("Predi regex", row)

            for pattern, replacement in patterns:
                row = re.sub(pattern, replacement, row, flags=re.IGNORECASE)

            print("Sled regex", row)

            try:
                print("@", row)
                parsed_data = eval(row)
                result = [[item[0], np.float32(item[1]), np.float32(item[2]), np.float32(item[3]), np.float32(item[4])]
                          for item in parsed_data]
                print("res", result)

            except Exception as e:
                print(f"Error parsing string: {e}")
                print(f"Problematic string: {row}")
                print(f"Problematic mask: {mask}")
                raise
            print("Sled try", result)
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
            print("Krai", flat_list, len(flat_list))
            capData.append(flat_list)


        capHnnData = []
        for row in df['hnnvoordinates']:
            print(row)
            row = row.replace("np.float32", "float")
            row = re.sub(r"float\((-?\d*\.?\d+(?:e[-+]?\d+)?)\)", r"\1", row)
            print("nachalo2")
            patterns = [
                (r"array\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*dtype=float32\)", r"\1"),
                (r"array\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*dtype=float64\)", r"\1"),
                (r"np\.float32\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)", r"\1"),
                (r"np\.float64\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)", r"\1"),
                (r"array\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)", r"\1"),
                (r"array\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*dtype=float32\)", r"\1"),
                (r"array\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*dtype=float64\)", r"\1"),
                (r"array\(([-+]?\d+\.)\s*,\s*dtype=float32\)", r"\1"),
                (r"array\(([-+]?\d+\.)\s*,\s*dtype=float64\)", r"\1"),
                (r"np\.float32\(([-+]?\d+\.)\s*\)", r"\1"),
                (r"np\.float64\(([-+]?\d+\.)\s*\)", r"\1"),
                (r"array\(([-+]?\d+\.)\s*\)", r"\1")
            ]

            print("Before regex", row)

            for pattern, replacement in patterns:
                row = re.sub(pattern, replacement, row, flags=re.IGNORECASE)
            print("After regex", row)


            try:
                # Parse the string into a Python list
                parsed_data = ast.literal_eval(row)
                print(parsed_data)
                # Convert float values to np.float32

                result = [[np.float32(item[0]), np.float32(item[1]), np.float32(item[2]), np.float32(item[3]), np.float32(item[4])]
                          for item in parsed_data]

            except ValueError as e:
                print(f"Error parsing string: {e}")
                print(f"Problematic string: {row}")
                raise
            print(result)
            cap = []
            index = 1
            for point in result:
                print("Point", point)
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
            capHnnData.append(flat_list)

        centerSizeData = []
        for row in df['coordinates']:
            parsed_data = ast.literal_eval(row)
            size = parsed_data[2] * parsed_data[3]
            centerX = parsed_data[0] + (parsed_data[2] / 2)
            centerY = parsed_data[1] + (parsed_data[3] / 2)
            print(size, centerX, centerY)
            centerSizeData.append([np.float32(size), np.float32(centerX), np.float32(centerY)])

        frameNumData = []
        for row in df['frameNum']:
            print(np.float32(row - 0.0001))
            frameNumData.append([np.float32(row - 0.0001)])

        scoreData = []
        for row in df['score']:
            print(row)
            scoreData.append(row)

        combined = []
        for l1, l2, l3, l4 in zip(capData, centerSizeData, frameNumData, capHnnData):
            print(l1, l2, l3, l4)
            combined.append(l1 + l2 + l3 + l4)
        print(combined[0], len(combined[0]))
        X_array = np.array(combined)
        print(X_array.shape)
        y_array = np.array(scoreData)
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
print(X.shape)
#print(len(data), data.head(5), data["coordinates"][0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network
model = Sequential([
    Dense(256, activation='relu', input_shape=(76,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(228, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(228, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(228, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Sigmoid for output between 0 and 1
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mean_squared_error',
              metrics=['mae'])

# Model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=75,
                    batch_size=32,
                    verbose=1)

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest MAE: {test_mae:.4f}")

# Example prediction
predictions = model.predict(X_test[:5])
print("\nSample predictions:", predictions.flatten())

# Save the model (optional)
model.save('nn_model.h5')

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', color='#1f77b4')
plt.plot(history.history['val_loss'], label='Validation Loss', color='#ff7f0e')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error Loss')
plt.legend()
plt.grid(True)
plt.show()