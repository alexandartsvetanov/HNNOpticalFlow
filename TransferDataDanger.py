import os
import re
import pandas as pd
from codeFromPaperHnn.Config import paths

#This script copies one column from a dataset to another dataset

path = paths['mainfolder']

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


for mask in get_mask_subdirs_os2(path):
    # File paths for input and output CSV files
    input_file = mask + "/trainDataHNN.csv"  # File to read the specific cell value from
    output_file = mask + "/trainData.csv"  # File to add the "Danger" column to

    # Read the input CSV file
    input_df = pd.read_csv(input_file)

    # Read the specific cell value (e.g., first row, first column)
    # Adjust row (0-based index) and column (name or index) as needed
    specific_cell_value = input_df.iloc[1, 5]  # Example: cell at row 1, column 1
    print(f"Value read from cell (row 1, column 1): {specific_cell_value}")

    # Read the existing output CSV file
    output_df = pd.read_csv(output_file)

    if 'danger' in output_df.columns:
        output_df = output_df.drop('danger', axis=1)
        print("Existing 'danger' column removed")

    # Add a new "Danger" column filled with the specific cell value
    # Only fill rows that have at least one non-null value in other columns
    output_df['danger'] = specific_cell_value

    # Save the modified DataFrame back to the output CSV
    output_df.to_csv(output_file, index=False)
    print(f"Modified CSV saved to {output_file}")