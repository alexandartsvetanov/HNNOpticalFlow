import torch, argparse
import numpy as np

import torch
from pathlib import Path
import os, sys
import pandas as pd
from numpy.ma.core import append
import seaborn as sns
from sympy import andre
from sympy.tensor import tensor

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)


import torch
import numpy as np

from argparse import Namespace
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
import re
import ast
import cv2

def visualize(x, y, vx, vy):

    # Create a blank image (white background, 400x400 pixels)
    img = np.ones((600, 400, 3), dtype=np.uint8) * 0

    # Draw the point as a circle
    point_color = (0, 0, 255)  # Red color in BGR
    radius = 5
    thickness = -1  # Filled circle
    cv2.circle(img, (int(x), int(y)), radius, point_color, thickness)

    # Draw the vector as an arrowed line
    vector_color = (0, 255, 0)  # Green color in BGR
    end_point = (int(x + 5* vx), int(y + 5* vy))
    cv2.arrowedLine(img, (int(x), int(y)), end_point, vector_color, 2, tipLength=0.2)

    # Display the image
    cv2.imshow('Point and Vector', img)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
def parse_custom_array(s):
    # Remove np.float32() wrappers while keeping the numeric values
    cleaned = re.sub(r'np\.float32\(([^)]+)\)', r'\1', s)
    # Convert to actual list
    data = ast.literal_eval(cleaned)
    # Convert numbers to float32 if neededparse_custom_array
    numpy_array = np.array([[x[0]] + [np.float32(y) for y in x[1:]] for x in data])
    return numpy_array

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

def getMyDataAvg():
    #print("vsichki:", get_mask_subdirs_os2(f'C:/Users/Alexs/PyCharmMiscProject/pythonProject7/object_tracking/object_tracking/samMooooolq/segmentAnything'))
    # Display the DataFrame
    x = torch.tensor([])
    dxdt = torch.tensor([])
    test_x = torch.tensor([])
    test_dxdt = torch.tensor([])
    allData = []
    i = 0
    for mask in get_mask_subdirs_os2(f'C:/Users/Alexs/PyCharmMiscProject/pythonProject7/object_tracking/object_tracking/samMooooolq/segmentAnything'):
        #if i > 0:
            #continue
        #i = i + 1
        print(mask)
        if Path(mask) == Path(
                f'C:/Users/Alexs/PyCharmMiscProject/pythonProject7/object_tracking/object_tracking/samMooooolq/segmentAnything/videos13/mask3'):
            continue
        df = pd.read_csv(mask + '/trainData.csv')
        #print(df.head())
        #print(df['cap'])

        rowindex = 0
        prevrow = []
        dataset = {}
        dataset['x'] = []
        dataset['dx'] = []
        for row in df['cap']:

            row = parse_custom_array(row)
            if row.size == 0:
                continue
            if rowindex < 7:
                prevrow.append(row)
                rowindex += 1
                continue
            i = 0
            j = 0
            indexes = []
            for itein in row:
                indexes.append(itein[0])
            #print("Row Index", row, indexes)
            countsListPrev = np.zeros(9)
            countsListCurrent = np.zeros(9)
            sumsListPrev = np.array([np.zeros(4) for _ in range(9)])
            sumsListCurrent = np.array([np.zeros(4) for _ in range(9)])
            for index, prevRow in enumerate(prevrow):
                if index < 4:
                    for item in prevRow:
                        if item[0] in indexes:
                            #print(int(item[0]) - 1)

                            sumsListPrev[int(item[0]) - 1] += item[1:]  # Add the slice (must match shape (4,))
                            countsListPrev[int(item[0]) - 1] += 1
                else:
                    for item in prevRow:
                        if item[0] in indexes:
                            #print(int(item[0]) - 1)
                            sumsListCurrent[int(item[0]) - 1] += item[1:]  # Add the slice (must match shape (4,))
                            countsListCurrent[int(item[0]) - 1] += 1
            for item in row:
                sumsListCurrent[int(item[0]) - 1] += item[1:]  # Add the slice (must match shape (4,))
                countsListCurrent[int(item[0]) - 1] += 1
            #print("sumsList", sumsList)
            #print("countsList", countsList)
            #sumsList = sumsList / countsList[:, np.newaxis]
            sumsListPrev = np.divide(sumsListPrev, countsListPrev[:, np.newaxis],
                                 out=np.full_like(sumsListPrev, np.nan),
                                 where=countsListPrev[:, np.newaxis] != 0)
            filled_arrPrev = np.nan_to_num(sumsListPrev, nan=0, posinf=1e10, neginf=-1e10)
            sumsListCurrent = np.divide(sumsListCurrent, countsListCurrent[:, np.newaxis],
                                 out=np.full_like(sumsListCurrent, np.nan),
                                 where=countsListCurrent[:, np.newaxis] != 0)
            filled_arrCurrent = np.nan_to_num(sumsListCurrent, nan=0, posinf=1e10, neginf=-1e10)
            print("sumsList1", filled_arrPrev[0])
            print("sumsList2", filled_arrCurrent[0])

            for ind in range(len(row)):
                index = int(indexes[ind])
                if index == 1:# and mask == "C:/Users/Alexs/PyCharmMiscProject/pythonProject7/object_tracking/object_tracking/samMooooolq/segmentAnything\\videos16\\mask1":
                    print(mask)
                    visualize(filled_arrCurrent[index - 1].tolist()[0], filled_arrCurrent[index - 1].tolist()[1], filled_arrCurrent[index - 1].tolist()[2], filled_arrCurrent[index - 1].tolist()[3])
                #print(index, ind)
                #print(filled_arr[index - 1].tolist())
                #print(row[ind][1:])
                dataset['x'].append(filled_arrPrev[index - 1].tolist())
                dataset['dx'].append(filled_arrCurrent[index - 1].tolist())
            #print("prevrow", prevrow[0][0], prevrow[1][0], prevrow[2][0], prevrow[4][0])
            prevrow.pop(0)
            prevrow.append(row)
            #print("prevrow", prevrow[0][0], prevrow[1][0], prevrow[2][0], prevrow[4][0])

    allData.append(dataset)



    print(len(x), x.shape, x[0])
    x_np = x.detach().numpy()
    mean = np.mean(x_np[:, 0])
    std = np.std(x_np[:, 0])
    maxx = np.max(x_np[:, 0])
    minn = np.min(x_np[:, 0])
    print(mean, std, maxx, minn)

    plt.figure(figsize=(10, 5))
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.hist(x_np[:, i], bins=30, color='skyblue', edgecolor='black')
        plt.title(f'Value {i}')
        plt.xlabel('Value Range')
        plt.ylabel('Number of Examples')

    plt.tight_layout()
    plt.show()

    #print(allData)

    all_x = []
    all_dx = []

    # Iterate through each dataset in allData (list of dicts)
    for dataset in allData:
        all_x.extend(dataset['x'])  # Extend with x arrays
        all_dx.extend(dataset['dx'])  # Extend with dx arrays

    # Convert to numpy arrays (shape: [N, 4])
    x_array = np.vstack(all_x)  # Stack all x values (N rows, 4 columns)
    dx_array = np.vstack(all_dx)  # Stack all dx values (N rows, 4 columns)

    first_column = [sublist[0] for sublist in x_array]
    max_value = max(first_column)
    print("Max na conv", max_value)

    plt.figure(figsize=(12, 8))
    for i in range(4):  # For each of the 4 positions
        plt.subplot(2, 2, i + 1)
        plt.hist(x_array[:, i], bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.title(f'Histogram of x (position {i + 1})')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    for i in range(4):  # For each of the 4 positions
        plt.subplot(2, 2, i + 1)
        plt.hist(dx_array[:, i], bins=20, alpha=0.7, color='red', edgecolor='black')
        plt.title(f'Histogram of dx (position {i + 1})')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    return x, dxdt, test_x, test_dxdt

getMyDataAvg()