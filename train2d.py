# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski
# da razbera kak da model trajectory of projectile with hsmiltonian!!!!!!!!!!!!!
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

from codeFromPaperHnn.nn_models import MLP
from codeFromPaperHnn.hnn import HNN
from codeFromPaperHnn.utils import L2_loss
from codeFromPaperHnn.data import get_dataset

'''from .nn_models import MLP
from .hnn import HNN
from .data import get_dataset
from .utils import L2_loss, rk4
'''
import torch
import numpy as np

from argparse import Namespace
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
import re
import ast


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=2, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=200, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=20, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='spring', type=str, help='only one option right now')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--use_rk4', dest='use_rk4', action='store_true', help='integrate derivative with RK4')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--field_type', default='solenoidal', type=str, help='type of vector field to learn')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.set_defaults(feature=True)
    return parser.parse_args()


#################################3
def parse_custom_array(s):
    # Remove np.float32() wrappers while keeping the numeric values
    cleaned = re.sub(r'np\.float32\(([^)]+)\)', r'\1', s)
    # Convert to actual list
    data = ast.literal_eval(cleaned)
    # Convert numbers to float32 if neededparse_custom_array
    numpy_array = np.array([[x[0]] + [np.float32(y) for y in x[1:]] for x in data])
    return numpy_array


def split_data(original_dict, test_ratio=0.2):
    """
    Splits dictionary data into training and test sets

    Args:
        original_dict: Dictionary with 'x' and 'dx' keys
        test_ratio: Proportion of data for test set (default 0.2)

    Returns:
        Dictionary with x, dx, testx, testdx keys
    """
    # Make sure we have arrays to work with
    x = np.array(original_dict['x'])
    dx = np.array(original_dict['dx'])

    # Calculate split index
    split_idx = int(len(x) * (1 - test_ratio))

    # Create new dictionary
    new_dict = {
        'x': x[:split_idx],  # 80% of original x
        'dx': dx[:split_idx],  # 80% of original dx
    }
    test_dict = {
        'x': x[split_idx:],  # 20% of original x
        'dx': dx[split_idx:],  # 20% of original dx
    }

    return new_dict, test_dict


# Read the Excel file

import math


def vector_angle_length(point_x, point_y, vector_x, vector_y):
    """
    Calculate the angle (in degrees) and length of a vector from given point and vector coordinates.

    Args:
        point_x (float): x-coordinate of the starting point
        point_y (float): y-coordinate of the starting point
        vector_x (float): x-component of the vector
        vector_y (float): y-component of the vector

    Returns:
        tuple: (angle in degrees, length of vector)
    """
    # Calculate vector length (magnitude)
    length = math.sqrt(vector_x ** 2 + vector_y ** 2)

    # Calculate angle in radians and convert to degrees
    # Using atan2 to handle all quadrants correctly
    angle_rad = math.atan2(vector_y, vector_x)
    angle_deg = math.degrees(angle_rad)

    # Normalize angle to 0-360 degrees
    angle_deg = (angle_deg % 360) / 360

    return (angle_deg, length)


def get_mask_subdirs_os(directory_path):
    if not os.path.isdir(directory_path):
        raise ValueError(f"'{directory_path}' is not a valid directory")

    pattern = re.compile(r'^mask\d+$')
    mask_subdirs = [
        os.path.join(directory_path, name)
        for name in os.listdir(directory_path)
        if os.path.isdir(os.path.join(directory_path, name)) and pattern.match(name)
    ]

    return sorted(mask_subdirs)


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


def hamiltonian_qp_to_xy(pa, q, r, vr, mass=1.0):
    """
    Convert Hamiltonian (q, p) + additional info back to Cartesian (x, y, vx, vy).

    Args:
        pa (float): Scaled angular momentum.
        q (float): Scaled angle θ.
        r_scaled (float): Scaled radial distance.
        vr_scaled (float): Scaled radial velocity.
        mass (float): Mass (default=1.0).

    Returns:
        x, y (float): Position.
        vx, vy (float): Velocity.
    """
    x_c, y_c = 225.0, 125.0

    # Rescale inputs

    theta = (q * 2 * np.pi) - np.pi  # Map [0, 1] back to [-π, π]
    # p_theta = 21000 + (pa + 1) * (71000 - 21000) / 2  # Map [-
    p_theta = pa * 2000 - 1000

    # Reconstruct position (x, y)
    x_rel = r * np.cos(theta)
    y_rel = r * np.sin(theta)
    x = x_rel + x_c
    y = y_rel + y_c

    # Reconstruct velocity (vx, vy)
    # vr = (x vx + y vy) / r
    # p_θ = m (x vy - y vx)
    # Solve the system:
    # [x_rel,  y_rel][vx] = [vr * r]
    # [-y_rel, x_rel][vy]   [p_θ / m]
    A = np.array([[x_rel, y_rel], [-y_rel, x_rel]])
    b = np.array([vr * r, p_theta / mass])
    vx, vy = np.linalg.solve(A, b)

    return x, y, vx, vy


def xy_to_hamiltonian_qp(x, y, vx, vy, mass=1.0):
    """
    Convert Cartesian coordinates (x, y, vx, vy) to Hamiltonian (q, p) in polar coordinates.

    Args:
        x, y (float): Particle position.
        vx, vy (float): Particle velocity components.
        mass (float): Particle mass (default=1.0).

    Returns:
        q (float): Generalized coordinate (angle theta in radians).
        p (float): Generalized momentum (angular momentum p_theta).
    """
    # x_c, y_c = 640.0, 360.0
    x_c, y_c = 225.0, 125.0
    # x_c, y_c = 0, 0
    x = x - x_c
    y = y - y_c
    # Generalized coordinate: angle θ
    q = np.arctan2(y, x)

    q_scaled = (q + np.pi) / (2 * np.pi)
    # q_scaled = q / np.pi

    # Generali zed momentum: angular momentum p_θ = m * (x * vy - y * vx)
    p = mass * (x * vy - y * vx)
    if p < 0 and p < -1000:
        p = -1000
    if p > 0 and p > 1000:
        p = 1000
    p_scaled = (p + 1000) / (2000)
    # 1280, 720
    r = np.sqrt(x ** 2 + y ** 2)

    # Compute radial velocity vr (needed for inversion)
    vr = (x * vx + y * vy) / r if r != 0 else 0.0
    # print("Dvata r", r, vr)
    # return [float(p_scaled), float(q_scaled)], [r, vr]
    return [float(p_scaled), float(q_scaled)], [r, vr]


def vector_to_angle_magnitude(x, y, vx, vy):
    """
    Convert point (x, y) and vector (vx, vy) to angle with x-axis (in degrees) and magnitude.
    Returns tuple (angle, magnitude).
    """
    magnitude = math.sqrt(vx ** 2 + vy ** 2)
    angle = math.degrees(math.atan2(vy, vx)) / 180.0

    max_magnitude = 20
    min_magnitude = 0
    mean_magnitude = 2.9
    std_magnitude = 4.5
    magnitude_range = max_magnitude - min_magnitude

    if magnitude > 18:
        magnitude = 18
    normalized_magnitude = (magnitude - mean_magnitude) / std_magnitude
    if normalized_magnitude > 2 * std_magnitude:
        normalized_magnitude = 2 * std_magnitude

    return (angle, normalized_magnitude)


def angle_magnitude_to_vector(x, y, angle, magnitude):
    """
    Convert point (x, y), normalized angle, and normalized magnitude back to vector (vx, vy).
    Uses mean and standard deviation for denormalization.
    Returns tuple (x, y, vx, vy).
    """
    # Denormalize angle: normalized * std + mean
    angle = angle * 180

    # Denormalize magnitude: normalized * std + mean
    mean_magnitude = 3.7
    std_magnitude = 9.3
    magnitude = magnitude * std_magnitude + mean_magnitude

    # Reconstruct vector components
    angle_rad = math.radians(angle)
    vx = magnitude * math.cos(angle_rad)
    vy = magnitude * math.sin(angle_rad)

    return (x, y, vx, vy)

def scaleData(x, y, vx, vy):
    x = x / 450
    y = y / 250
    if vx < -12:
        vx = -12
    if vy < -12:
        vy = -12
    if vx > 12:
        vx = 12
    if vy > 12:
        vy = 12
    vx = (vx + 12) / 24
    vy = (vy + 12) / 24

    return [[x, y], [vx, vy]]


def unscaleData(x, y, vx, vy):
    # Reverse position scaling (x, y)
    x = x * 450
    y = y * 250

    # Reverse velocity scaling (vx, vy)
    vx = vx * 24 - 12
    vy = vy * 24 - 12

    return x, y, vx, vy

def scale_first_elements(dictionary):
    # Extract all first elements from lists in both keys
    first_elements = []
    for key in ['x', 'dx']:
        for item in dictionary[key]:
            first_elements.append(item[0])

    # Find min and max for scaling
    min_val = min(first_elements)
    max_val = max(first_elements)
    print("Min max", min_val, max_val)  # -3311.7173 401.24118
    range_val = max_val - min_val if max_val != min_val else 1  # Avoid division by zero

    # Create new dictionary with scaled values
    scaled_dict = {}
    for key in ['x', 'dx']:
        scaled_dict[key] = [
            [((item[0] - min_val) / range_val), item[1]] for item in dictionary[key]
        ]

    return scaled_dict


def scale_first_elementsT(tensor):
    # Extract all first elements from the tensor
    first_elements = tensor[:, 0].flatten().tolist()

    # Find min and max for scaling
    min_val = min(first_elements)
    max_val = max(first_elements)
    print("Min max", min_val, max_val)  # -3311.7173 401.24118
    range_val = max_val - min_val if max_val != min_val else 1  # Avoid division by zero

    # Create new tensor with scaled first elements
    scaled_tensor = torch.zeros_like(tensor)
    scaled_tensor[:, 0] = (tensor[:, 0] - min_val) / range_val
    scaled_tensor[:, 1] = tensor[:, 1]

    return scaled_tensor


def getMyDataAvg():
    print("vsichki:", get_mask_subdirs_os2(
        f'C:/Users/Alexs/PyCharmMiscProject/pythonProject7/object_tracking/object_tracking/samMooooolq/segmentAnything'))
    # Display the DataFrame
    x = torch.tensor([])
    dxdt = torch.tensor([])
    test_x = torch.tensor([])
    test_dxdt = torch.tensor([])
    allData = []
    i = 0
    for mask in get_mask_subdirs_os2(
            f'C:/Users/Alexs/PyCharmMiscProject/pythonProject7/object_tracking/object_tracking/samMooooolq/segmentAnything'):
        # if i > 0:
        # continue
        # i = i + 1
        print(mask)
        if Path(mask) == Path(
                f'C:/Users/Alexs/PyCharmMiscProject/pythonProject7/object_tracking/object_tracking/samMooooolq/segmentAnything/videos13/mask3'):
            continue
        df = pd.read_csv(mask + '/trainData.csv')
        # print(df.head())
        # print(df['cap'])

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
            print("Row Index", row, indexes)
            countsListPrev = np.zeros(9)
            countsListCurrent = np.zeros(9)
            sumsListPrev = np.array([np.zeros(4) for _ in range(9)])
            sumsListCurrent = np.array([np.zeros(4) for _ in range(9)])
            for index, prevRow in enumerate(prevrow):
                if index < 4:
                    for item in prevRow:
                        if item[0] in indexes:
                            # print(int(item[0]) - 1)
                            sumsListPrev[int(item[0]) - 1] += item[1:]  # Add the slice (must match shape (4,))
                            countsListPrev[int(item[0]) - 1] += 1
                else:
                    for item in prevRow:
                        if item[0] in indexes:
                            # print(int(item[0]) - 1)
                            sumsListCurrent[int(item[0]) - 1] += item[1:]  # Add the slice (must match shape (4,))
                            countsListCurrent[int(item[0]) - 1] += 1
            for item in row:
                sumsListCurrent[int(item[0]) - 1] += item[1:]  # Add the slice (must match shape (4,))
                countsListCurrent[int(item[0]) - 1] += 1
            # print("sumsList", sumsList)
            # print("countsList", countsList)
            # sumsList = sumsList / countsList[:, np.newaxis]
            sumsListPrev = np.divide(sumsListPrev, countsListPrev[:, np.newaxis],
                                     out=np.full_like(sumsListPrev, np.nan),
                                     where=countsListPrev[:, np.newaxis] != 0)
            filled_arrPrev = np.nan_to_num(sumsListPrev, nan=0, posinf=1e10, neginf=-1e10)
            sumsListCurrent = np.divide(sumsListCurrent, countsListCurrent[:, np.newaxis],
                                        out=np.full_like(sumsListCurrent, np.nan),
                                        where=countsListCurrent[:, np.newaxis] != 0)
            filled_arrCurrent = np.nan_to_num(sumsListCurrent, nan=0, posinf=1e10, neginf=-1e10)
            print("sumsList1", filled_arrPrev)
            print("sumsList2", filled_arrCurrent)

            for ind in range(len(row)):
                index = int(indexes[ind])
                # print(index, ind)
                # print(filled_arr[index - 1].tolist())
                # print(row[ind][1:])
                dataset['x'].append(filled_arrPrev[index - 1].tolist())
                dataset['dx'].append(filled_arrCurrent[index - 1].tolist())
            # print("prevrow", prevrow[0][0], prevrow[1][0], prevrow[2][0], prevrow[4][0])
            prevrow.pop(0)
            prevrow.append(row)
            # print("prevrow", prevrow[0][0], prevrow[1][0], prevrow[2][0], prevrow[4][0])

        allData.append(dataset)

        split_dict, test_dataset = split_data(dataset)
        # print(len(split_dict['x']), len(test_dataset['x']), len(split_dict['dx']), len(test_dataset['dx']), split_dict['x'][0])
        m = max(max(sublist) for sublist in sum(dataset.values(), []))
        print("Max", m)

        #split_dict['x'] = [xy_to_hamiltonian_qp(row[0], row[1], row[2], row[3])[0] for row in split_dict['x']]
        #test_dataset['x'] = [xy_to_hamiltonian_qp(row[0], row[1], row[2], row[3])[0] for row in test_dataset['x']]
        #split_dict['dx'] = [xy_to_hamiltonian_qp(row[0], row[1], row[2], row[3])[0] for row in split_dict['dx']]
        #test_dataset['dx'] = [xy_to_hamiltonian_qp(row[0], row[1], row[2], row[3])[0] for row in test_dataset['dx']]

        split_dict['x'] = [[scaleData(row[0], row[1], row[2], row[3])] for row in split_dict['x']]
        test_dataset['x'] = [[scaleData(row[0], row[1], row[2], row[3])] for row in test_dataset['x']]
        split_dict['dx'] = [[scaleData(row[0], row[1], row[2], row[3])] for row in split_dict['dx']]
        test_dataset['dx'] = [[scaleData(row[0], row[1], row[2], row[3])] for row in test_dataset['dx']]

        print(len(split_dict['x']), len(test_dataset['x']), len(split_dict['dx']), len(test_dataset['dx']),
              split_dict['x'][0])

        x = torch.cat((x, torch.tensor(split_dict['x'], requires_grad=True, dtype=torch.float32)), dim=0)
        dxdt = torch.cat((dxdt, torch.tensor(split_dict['dx'], requires_grad=True, dtype=torch.float32)), dim=0)
        test_x = torch.cat((test_x, torch.tensor(test_dataset['x'], requires_grad=True, dtype=torch.float32)), dim=0)
        test_dxdt = torch.cat((test_dxdt, torch.tensor(test_dataset['dx'], requires_grad=True, dtype=torch.float32)),
                              dim=0)

    print(len(x), x.shape, x[0])
    x_np = x.detach().numpy().squeeze(1)

    '''mean = np.mean(x_np[:, 2])
    std = np.std(x_np[:, 2])
    maxx = np.max(x_np[:, 2])
    minn = np.min(x_np[:, 2])
    print(mean, std, maxx, minn)'''
    x_np = x_np.reshape(15066, 4)
    print(x_np.shape, x_np[:, 0].shape)

    plt.figure(figsize=(10, 5))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.hist(x_np[:, i], bins=30, color='skyblue', edgecolor='black')
        plt.title(f'Value {i}')
        plt.xlabel('Value Range')
        plt.ylabel('Number of Examples')

    plt.tight_layout()
    plt.show()

    # print(allData)
    '''
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
    plt.show()'''

    return x, dxdt, test_x, test_dxdt


def getMyDataRefined():
    print("vsichki:", get_mask_subdirs_os2(
        f'C:/Users/Alexs/PyCharmMiscProject/pythonProject7/object_tracking/object_tracking/samMooooolq/segmentAnything'))
    # Display the DataFrame
    x = torch.tensor([])
    dxdt = torch.tensor([])
    test_x = torch.tensor([])
    test_dxdt = torch.tensor([])
    allData = []
    i = 0
    for mask in get_mask_subdirs_os2(
            f'C:/Users/Alexs/PyCharmMiscProject/pythonProject7/object_tracking/object_tracking/samMooooolq/segmentAnything'):
        # if i > 0:
        # continue
        # i = i + 1
        print(mask)
        df = pd.read_csv(mask + '/trainData.csv')
        # print(df.head())
        # print(df['cap'])

        rowindex = 0
        prevrow = []
        dataset = {}
        dataset['x'] = []
        dataset['dx'] = []
        for row in df['cap']:

            row = parse_custom_array(row)
            if row.size == 0:
                continue
            if rowindex < 5:
                prevrow.append(row)
                rowindex += 1
                continue
            i = 0
            j = 0
            indexes = []
            for itein in row:
                indexes.append(itein[0])
            print("Row Index", row, indexes)
            countsList = np.zeros(9)
            sumsList = np.array([np.zeros(4) for _ in range(9)])
            for prevRow in prevrow:
                for item in prevRow:
                    if item[0] in indexes:
                        # print(int(item[0]) - 1)
                        sumsList[int(item[0]) - 1] += item[1:]  # Add the slice (must match shape (4,))
                        countsList[int(item[0]) - 1] += 1
            # print("sumsList", sumsList)
            # print("countsList", countsList)
            # sumsList = sumsList / countsList[:, np.newaxis]
            sumsList = np.divide(sumsList, countsList[:, np.newaxis],
                                 out=np.full_like(sumsList, np.nan),
                                 where=countsList[:, np.newaxis] != 0)
            filled_arr = np.nan_to_num(sumsList, nan=0, posinf=1e10, neginf=-1e10)
            # print("sumsList2", filled_arr)
            for ind in range(len(row)):
                index = int(indexes[ind])
                # print(index, ind)
                # print(filled_arr[index - 1].tolist())
                # print(row[ind][1:])
                dataset['x'].append(filled_arr[index - 1].tolist())
                dataset['dx'].append(row[ind][1:])
            # print("prevrow", prevrow[0][0], prevrow[1][0], prevrow[2][0], prevrow[4][0])
            prevrow.pop(0)
            prevrow.append(row)
            # print("prevrow", prevrow[0][0], prevrow[1][0], prevrow[2][0], prevrow[4][0])

        allData.append(dataset)

        split_dict, test_dataset = split_data(dataset)
        # print(len(split_dict['x']), len(test_dataset['x']), len(split_dict['dx']), len(test_dataset['dx']), split_dict['x'][0])
        m = max(max(sublist) for sublist in sum(dataset.values(), []))
        print("Max", m)

        split_dict['x'] = [xy_to_hamiltonian_qp(row[0], row[1], row[2], row[3])[0] for row in split_dict['x']]
        test_dataset['x'] = [xy_to_hamiltonian_qp(row[0], row[1], row[2], row[3])[0] for row in test_dataset['x']]
        split_dict['dx'] = [xy_to_hamiltonian_qp(row[0], row[1], row[2], row[3])[0] for row in split_dict['dx']]
        test_dataset['dx'] = [xy_to_hamiltonian_qp(row[0], row[1], row[2], row[3])[0] for row in test_dataset['dx']]

        print(len(split_dict['x']), len(test_dataset['x']), len(split_dict['dx']), len(test_dataset['dx']),
              split_dict['x'][0])

        x = torch.cat((x, torch.tensor(split_dict['x'], requires_grad=True, dtype=torch.float32)), dim=0)
        dxdt = torch.cat((dxdt, torch.tensor(split_dict['dx'], requires_grad=True, dtype=torch.float32)), dim=0)
        test_x = torch.cat((test_x, torch.tensor(test_dataset['x'], requires_grad=True, dtype=torch.float32)), dim=0)
        test_dxdt = torch.cat((test_dxdt, torch.tensor(test_dataset['dx'], requires_grad=True, dtype=torch.float32)),
                              dim=0)

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

    # print(allData)

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


# x, dxdt, test_x, test_dxdt = getMyDataAvg()
def getMyData():
    print("vsichki:", get_mask_subdirs_os2(
        f'C:/Users/Alexs/PyCharmMiscProject/pythonProject7/object_tracking/object_tracking/samMooooolq/segmentAnything'))
    # Display the DataFrame
    x = torch.tensor([])
    dxdt = torch.tensor([])
    test_x = torch.tensor([])
    test_dxdt = torch.tensor([])
    allData = []
    i = 0
    for mask in get_mask_subdirs_os2(
            f'C:/Users/Alexs/PyCharmMiscProject/pythonProject7/object_tracking/object_tracking/samMooooolq/segmentAnything'):
        # if i > 0:
        # continue
        # i = i + 1
        print(mask)
        df = pd.read_csv(mask + '/trainData.csv')
        # print(df.head())
        # print(df['cap'])

        rowindex = 0
        prevrow = []
        dataset = {}
        dataset['x'] = []
        dataset['dx'] = []
        for row in df['cap']:
            row = parse_custom_array(row)
            if rowindex == 0:
                # print(f"Raw row content: {row!r}")
                prevrow = row
                rowindex += 1
                # print("prev", prevrow, type(prevrow))
                continue
            i = 0
            j = 0
            while i < len(prevrow) and j < len(row):
                if prevrow[i][0] == row[j][0]:
                    dataset['x'].append(prevrow[i][1:])
                    dataset['dx'].append(row[j][1:])
                    print("X", prevrow[i][1:])
                    print("DX", row[j][1:])
                    i = i + 1
                    j = j + 1
                    continue
                if prevrow[i][0] < row[j][0] and i < len(prevrow):
                    i = i + 1
                    continue
                if prevrow[i][0] > row[j][0] and j < len(row):
                    j = j + 1
                    continue
            prevrow = row
        # print(dataset)

        # Function to encode a list of values

        # dataEncoded = encode_pairs(dataset)
        allData.append(dataset)
        split_dict, test_dataset = split_data(dataset)
        # print(len(split_dict['x']), len(test_dataset['x']), len(split_dict['dx']), len(test_dataset['dx']), split_dict['x'][0])
        m = max(max(sublist) for sublist in sum(dataset.values(), []))

        print("Max", m)
        # split_dict['x'] = [[row[0] * m + row[1], row[2] * 10 + row[3]] for row in split_dict['x']]
        # test_dataset['x'] = [[row[0] * m + row[1], row[2] * 10 + row[3]] for row in test_dataset['x']]
        # split_dict['dx'] = [[row[0] * m + row[1], row[2] * 10 + row[3]] for row in split_dict['dx']]
        # test_dataset['dx'] = [[row[0] * m + row[1], row[2] * 10 + row[3]] for row in test_dataset['dx']]
        '''
        split_dict['x'] = [[row[0] / 1280, row[1] / 720, row[2] / 9.7, row[3] / 14.7] for row in split_dict['x']]
        test_dataset['x'] = [[row[0] / 1280, row[1] / 720, row[2] / 9.7, row[3] / 14.7] for row in test_dataset['x']]
        split_dict['dx'] = [[row[0] / 1280, row[1] / 720, row[2] / 9.7, row[3] / 14.7] for row in split_dict['dx']]
        test_dataset['dx'] = [[row[0] / 1280, row[1] / 720, row[2] / 9.7, row[3] / 14.7] for row in test_dataset['dx']]
        '''

        split_dict['x'] = [xy_to_hamiltonian_qp(row[0], row[1], row[2], row[3])[0] for row in split_dict['x']]
        test_dataset['x'] = [xy_to_hamiltonian_qp(row[0], row[1], row[2], row[3])[0] for row in test_dataset['x']]
        split_dict['dx'] = [xy_to_hamiltonian_qp(row[0], row[1], row[2], row[3])[0] for row in split_dict['dx']]
        test_dataset['dx'] = [xy_to_hamiltonian_qp(row[0], row[1], row[2], row[3])[0] for row in test_dataset['dx']]

        # split_dict['x'] = [vector_to_angle_magnitude(row[0], row[1], row[2], row[3]) for row in split_dict['x']]
        # test_dataset['x'] = [vector_to_angle_magnitude(row[0], row[1], row[2], row[3]) for row in test_dataset['x']]
        # split_dict['dx'] = [vector_to_angle_magnitude(row[0], row[1], row[2], row[3]) for row in split_dict['dx']]
        # test_dataset['dx'] = [vector_to_angle_magnitude(row[0], row[1], row[2], row[3]) for row in test_dataset['dx']]

        print(len(split_dict['x']), len(test_dataset['x']), len(split_dict['dx']), len(test_dataset['dx']),
              split_dict['x'][0])

        x = torch.cat((x, torch.tensor(split_dict['x'], requires_grad=True, dtype=torch.float32)), dim=0)
        dxdt = torch.cat((dxdt, torch.tensor(split_dict['dx'], requires_grad=True, dtype=torch.float32)), dim=0)
        test_x = torch.cat((test_x, torch.tensor(test_dataset['x'], requires_grad=True, dtype=torch.float32)), dim=0)
        test_dxdt = torch.cat((test_dxdt, torch.tensor(test_dataset['dx'], requires_grad=True, dtype=torch.float32)),
                              dim=0)

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

    # print(allData)

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


#####################################


globalModel = None


def train(args):
    global globalModel
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # init model and optimizer
    if args.verbose:
        print("Training baseline model:" if args.baseline else "Training HNN model:")

    output_dim = args.input_dim if args.baseline else 2
    #nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    #model = HNN(args.input_dim, differentiable_model=nn_model, field_type=args.field_type, baseline=args.baseline)
    nn_model = MLP(input_dim=2, hidden_dim=args.hidden_dim, output_dim=2, nonlinearity=args.nonlinearity)
    model = HNN(input_dim=2, differentiable_model=nn_model, field_type=args.field_type, baseline=2)
    globalModel = model
    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)

    # arrange data
    data = get_dataset(seed=args.seed)
    # print(data)
    x = torch.tensor(data['x'], requires_grad=True, dtype=torch.float32)
    test_x = torch.tensor(data['test_x'], requires_grad=True, dtype=torch.float32)
    dxdt = torch.Tensor(data['dx'])
    test_dxdt = torch.Tensor(data['test_dx'])
    x, dxdt, test_x, test_dxdt = getMyDataAvg()

    # vanilla train loop
    stats = {'train_loss': [], 'test_loss': []}
    for step in range(args.total_steps + 1):

        # train step
        dxdt_hat = model.rk4_time_derivative(x, dxdt) if args.use_rk4 else model.time_derivative(x)
        loss = L2_loss(dxdt, dxdt_hat)
        loss.backward();
        optim.step();
        optim.zero_grad()

        # run test data
        test_dxdt_hat = model.rk4_time_derivative(test_x, test_dxdt) if args.use_rk4 else model.time_derivative(test_x)
        test_loss = L2_loss(test_dxdt, test_dxdt_hat)

        # logging
        stats['train_loss'].append(loss.item())
        stats['test_loss'].append(test_loss.item())
        if args.verbose and step % args.print_every == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))

    train_dxdt_hat = model.time_derivative(x)
    train_dist = (dxdt - train_dxdt_hat) ** 2
    test_dxdt_hat = model.time_derivative(test_x)
    test_dist = (test_dxdt - test_dxdt_hat) ** 2
    print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
          .format(train_dist.mean().item(), train_dist.std().item() / np.sqrt(train_dist.shape[0]),
                  test_dist.mean().item(), test_dist.std().item() / np.sqrt(test_dist.shape[0])))

    return model, stats


if __name__ == "__main__":
    args = get_args()
    print(args, args.use_rk4)
    args.baseline = False
    args.verbose = True
    model, stats = train(args)
    # save
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    label = '-baseline' if args.baseline else '-hnn'
    label = '-rk4' + label if args.use_rk4 else label
    path = '{}/{}{}.tar'.format(args.save_dir, args.name, label)
    print(path)
    torch.save(model.state_dict(), path)
    plt.plot(stats['train_loss'], label='Train Loss')
    plt.plot(stats['test_loss'], label='Test Loss')
    # plt.ylim(0, 1.0)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    '''a =[250.6, 127.4, 0.13, -0.02]#, xy_to_hamiltonian_qp(100, 100, 0.7, 0.7)

    torch_tensor = torch.tensor(a, requires_grad=True)
    torch_tensor = torch_tensor.reshape(1, 4)
    print("Posledno", torch_tensor)
    dxdt_hat = globalModel.time_derivative(torch_tensor)
    print(dxdt_hat)
    val1, val2, val3, val4 = dxdt_hat[0]
    print("Novi", val1 * 1280, val2 * 720, val3 * 9.7, val4 * 14.7)'''

    a=[206, 40, -3, -1]  # , xy_to_hamiltonian_qp(100, 100, 0.7, 0.7)
    a=[406, 140, 3, -1]  # , xy_to_hamiltonian_qp(100, 100, 0.7, 0.7)
    torch_tensor = torch.tensor(a, requires_grad=True, dtype=torch.float32)
    torch_tensor = torch_tensor.reshape(2, 2)
    print("Posledno", torch_tensor)
    dxdt_hat = globalModel.time_derivative(torch_tensor)
    print(dxdt_hat)
    val1, val2 = dxdt_hat[0]  # Unpacks first row
    val3, val4 = dxdt_hat[1]  # Unpacks second row
    print(val1.item() * 450, val2.item() * 250, val3.item(), val4.item())

    dxdt_hat2 = globalModel.time_derivative(dxdt_hat)
    dxdt_hat3 = globalModel.time_derivative(dxdt_hat2)
    dxdt_hat4 = globalModel.time_derivative(dxdt_hat3)
    dxdt_hat5 = globalModel.time_derivative(dxdt_hat4)
    dxdt_hat6 = globalModel.time_derivative(dxdt_hat5)

    print("##################33")
    val1, val2 = dxdt_hat[0]  # Unpacks first row
    val3, val4 = dxdt_hat[1]  # Unpacks second row
    print(val1.item() * 450, val2.item() * 250, val3.item(), val4.item())

    print("##################33")
    val1, val2 = dxdt_hat[0]  # Unpacks first row
    val3, val4 = dxdt_hat[1]  # Unpacks second row
    print(val1.item() * 450, val2.item() * 250, val3.item(), val4.item())

    print("##################33")
    val1, val2 = dxdt_hat[0]  # Unpacks first row
    val3, val4 = dxdt_hat[1]  # Unpacks second row
    print(val1.item() * 450, val2.item() * 250, val3.item(), val4.item())

    print("##################33")
    val1, val2 = dxdt_hat[0]  # Unpacks first row
    val3, val4 = dxdt_hat[1]  # Unpacks second row
    print(val1.item() * 450, val2.item() * 250, val3.item(), val4.item())

    print("##################33")
    val1, val2 = dxdt_hat[0]  # Unpacks first row
    val3, val4 = dxdt_hat[1]  # Unpacks second row
    print(val1.item() * 450, val2.item() * 250, val3.item(), val4.item())


    '''a = vector_to_angle_magnitude(350, 200, -10, -5)  # , xy_to_hamiltonian_qp(100, 100, 0.7, 0.7)
    print("AAAA", a)
    b = angle_magnitude_to_vector(350, 200, a[0], a[1])
    print(b)
    torch_tensor = torch.tensor(a, requires_grad=True)
    torch_tensor = torch_tensor.reshape(1, 2)
    print("Posledno", torch_tensor)
    dxdt_hat = globalModel.time_derivative(torch_tensor)
    print(dxdt_hat)
    val1, val2 = dxdt_hat[0]
    bdx = angle_magnitude_to_vector(350, 200, val1.item(), val2.item())
    print(bdx)'''