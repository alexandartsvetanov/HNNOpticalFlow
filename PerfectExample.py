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


# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch, argparse
import numpy as np

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from nn_models import MLP
from hnn import HNN
from data2 import get_dataset
from utils import L2_loss, to_pickle, from_pickle

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=3*4, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=600, type=int, help='batch_size')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=10000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='3body', type=str, help='only one option right now')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--field_type', default='solenoidal', type=str, help='type of vector field to learn')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.set_defaults(feature=True)
    return parser.parse_args()

def train(args):
  # set random seed
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  # init model and optimizer
  if args.verbose:
    print("Training baseline model:" if args.baseline else "Training HNN model:")

  output_dim = args.input_dim if args.baseline else 2
  nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
  model = HNN(args.input_dim, differentiable_model=nn_model,
            field_type=args.field_type, baseline=args.baseline)
  optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)

  # arrange data
  data = get_dataset(args.name, args.save_dir, verbose=True)

  x = torch.tensor( data['coords'], requires_grad=True, dtype=torch.float32)
  test_x = torch.tensor( data['test_coords'], requires_grad=True, dtype=torch.float32)
  dxdt = torch.Tensor(data['dcoords'])
  test_dxdt = torch.Tensor(data['test_dcoords'])
  print(x.shape, dxdt.shape, test_dxdt.shape, test_x.shape)
  a= 1 / 0


args = get_args()
model, stats = train(args)


def scale_columns_neg1_pos1(tensor):
    # Compute min and max for each column (dim=0)
    col_mins = tensor.min(dim=0).values  # shape (M,)
    col_maxs = tensor.max(dim=0).values  # shape (M,)

    # Avoid division by zero (if a column is constant)
    zero_range_mask = (col_maxs == col_mins)
    col_maxs[zero_range_mask] = col_mins[zero_range_mask] + 1  # prevent NaN

    # Scale each column to [-1, 1]
    scaled_tensor = 2 * (tensor - col_mins) / (col_maxs - col_mins) - 1
    return scaled_tensor, col_mins, col_maxs

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=4, type=int, help='dimensionality of input tensor')
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


import numpy as np


def split_data_with_shuffle(data, test_size=0.2, random_state=None):
    """
    Splits a dictionary with 'x' and 'dx' keys into train and test sets with shuffling.

    Parameters:
    - data: Dictionary with keys 'x' and 'dx' containing data and labels
    - test_size: Proportion of data to include in test split (default 0.2)
    - random_state: Seed for random shuffling (optional)

    Returns:
    - trainData: Dictionary with training data
    - testData: Dictionary with test data
    """
    # Check if required keys exist
    if 'x' not in data or 'dx' not in data:
        raise ValueError("Input dictionary must contain 'x' and 'dx' keys")

    # Check data lengths match
    if len(data['x']) != len(data['dx']):
        raise ValueError("Length of 'x' and 'dx' must be equal")

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Get total number of samples
    n_samples = len(data['x'])

    # Create shuffled indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # Calculate split point
    split_idx = int(n_samples * (1 - test_size))

    # Split indices into train and test
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    # Create train and test dictionaries
    trainData = {
        'x': np.array(data['x'])[train_indices],
        'dx': np.array(data['dx'])[train_indices]
    }

    testData = {
        'x': np.array(data['x'])[test_indices],
        'dx': np.array(data['dx'])[test_indices]
    }

    return trainData, testData
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


import random


def generateSynteticData(n):
    dataset = {}
    dataset['x'] = []
    dataset['dx'] = []

    for _ in range(n):
        # Generate first list
        list1 = [
            random.randint(50, 400),  # First number: 50-400
            random.randint(50, 200),  # Second number: 50-200
            random.randint(5, 15),  # Third number: -15 to 15
            random.randint(5, 15)  # Fourth number: -15 to 15
        ]

        # Generate second list based on first list
        list2 = [
            list1[0] + list1[2] * random.uniform(2, 3),  # First number calculation
            list1[1] + list1[3] * random.uniform(2, 3),  # Second number calculation
            list1[2] * random.uniform(20, 20.5),  # Third number calculation
            list1[3] * random.uniform(20, 20.5)  # Fourth number calculation
        ]

        dataset['x'].append(list1)
        dataset['dx'].append(list2)

    return dataset


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
        if Path(mask) != Path(
                f'C:/Users/Alexs/PyCharmMiscProject/pythonProject7/object_tracking/object_tracking/samMooooolq/segmentAnything/videos16/mask1'):
            continue
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
            print("sumsList1", filled_arrPrev[0])
            print("sumsList2", filled_arrCurrent[0])

            for ind in range(len(row)):
                index = int(indexes[ind])
                print(filled_arrPrev[index - 1].tolist())
                print(filled_arrCurrent[index - 1].tolist())
                print([filled_arrCurrent[index - 1].tolist()[0], filled_arrCurrent[index - 1].tolist()[1],
                       (filled_arrCurrent[index - 1].tolist()[0] - filled_arrPrev[index - 1].tolist()[0]) / 0.1,
                       (filled_arrCurrent[index - 1].tolist()[1] - filled_arrPrev[index - 1].tolist()[1]) / 0.1])
                print([ (filled_arrCurrent[index - 1].tolist()[0] - filled_arrPrev[index - 1].tolist()[0]) / 0.1,
                       (filled_arrCurrent[index - 1].tolist()[1] - filled_arrPrev[index - 1].tolist()[1]) / 0.1,
                        filled_arrCurrent[index - 1].tolist()[2], filled_arrCurrent[index - 1].tolist()[3]])
                dataset['x'].append([[filled_arrCurrent[index - 1].tolist()[0], filled_arrCurrent[index - 1].tolist()[1]],
                                     [(filled_arrCurrent[index - 1].tolist()[0] - filled_arrPrev[index - 1].tolist()[0]) / 0.1,
                                      (filled_arrCurrent[index - 1].tolist()[1] - filled_arrPrev[index - 1].tolist()[1]) / 0.1]])
                dataset['dx'].append([[(filled_arrCurrent[index - 1].tolist()[0] - filled_arrPrev[index - 1].tolist()[0]) / 0.1,
                                       (filled_arrCurrent[index - 1].tolist()[1] - filled_arrPrev[index - 1].tolist()[1]) / 0.1],
                                       [filled_arrCurrent[index - 1].tolist()[2], filled_arrCurrent[index - 1].tolist()[3]]])

            # print("prevrow", prevrow[0][0], prevrow[1][0], prevrow[2][0], prevrow[4][0])
            prevrow.pop(0)
            prevrow.append(row)
            # print("prevrow", prevrow[0][0], prevrow[1][0], prevrow[2][0], prevrow[4][0])


        allData.append(dataset)
        for data in allData:
            for key in data:
                print(key)
                for pair_list in data[key]:
                    # Check if the pair_list has at least two elements (assuming it's always pairs)
                    if len(pair_list) >= 2:
                        # Iterate through each element in the second list of the pair
                        for i in range(len(pair_list[1])):
                            if pair_list[1][i] > 200 and key =="x":
                                pair_list[1][i] = 200
                            if pair_list[1][i] < -200 and key =="x":
                                pair_list[1][i] = -200
                            if pair_list[0][i] > 200 and key =="dx":
                                pair_list[0][i] = 200
                            if pair_list[0][i] < -200 and key =="dx":
                                pair_list[0][i] = -200
                            if pair_list[1][i] > 20 and key =="dx":
                                pair_list[1][i] = 20
                            if pair_list[1][i] < -20 and key =="dx":
                                pair_list[1][i] = -20

        print("Real data", dataset['x'][0], dataset['dx'][0])

        split_dict, test_dataset = split_data_with_shuffle(dataset)
        #split_dict, test_dataset = split_data(dataset)
        print(split_dict['x'][0], split_dict['dx'][0])
        # print(len(split_dict['x']), len(test_dataset['x']), len(split_dict['dx']), len(test_dataset['dx']), split_dict['x'][0])
        m = max(max(sublist) for sublist in sum(dataset.values(), []))
        print("Max", m)

        '''split_dict['x'] = [xy_to_hamiltonian_qp(row[0], row[1], row[2], row[3])[0] for row in split_dict['x']]
        test_dataset['x'] = [xy_to_hamiltonian_qp(row[0], row[1], row[2], row[3])[0] for row in test_dataset['x']]
        split_dict['dx'] = [xy_to_hamiltonian_qp(row[0], row[1], row[2], row[3])[0] for row in split_dict['dx']]
        test_dataset['dx'] = [xy_to_hamiltonian_qp(row[0], row[1], row[2], row[3])[0] for row in test_dataset['dx']]'''

        print(len(split_dict['x']), len(test_dataset['x']), len(split_dict['dx']), len(test_dataset['dx']),
              split_dict['x'][0], split_dict['dx'][0], test_dataset['x'][0], test_dataset['dx'][0])

        x = torch.cat((x, torch.tensor(split_dict['x'], requires_grad=True, dtype=torch.float32)), dim=0)
        #x = x.reshape(627, 4)
        print("molq shape", x.shape)
        dxdt = torch.cat((dxdt, torch.tensor(split_dict['dx'], requires_grad=True, dtype=torch.float32)), dim=0)
        test_x = torch.cat((test_x, torch.tensor(test_dataset['x'], requires_grad=True, dtype=torch.float32)), dim=0)
        test_dxdt = torch.cat((test_dxdt, torch.tensor(test_dataset['dx'], requires_grad=True, dtype=torch.float32)),
                              dim=0)
    transformed_tensor = x.tolist()

    transformed_tensor = [[a[0].tolist(), a[1].tolist()] for a in x]
    flattened_tensor = x.view(x.shape[0], 4)
    print("My shape",x.shape, x.dim(), x.shape[1])

    x_list = [[row[0].tolist(), row[1].tolist()] for row in x]
    dxdt_list = [[row[0].tolist(), row[1].tolist()] for row in dxdt]
    test_x_list = [[row[0].tolist(), row[1].tolist()] for row in test_x]
    test_dxdt_list = [[row[0].tolist(), row[1].tolist()] for row in test_dxdt]

    print(len(x_list))  # 627
    print(len(x_list[0]))  # 2 (each is a list of 2 elements)
    print(x_list[0][0])  # e.g., [0.5, -1.2] (first vector)
    print(x_list[0][1])  # e.g., [1.3, 0.8]  (second vector)

    x_np = x.detach().numpy()
    x_np2 = dxdt.detach().numpy()

    # Reshape the data for plotting (flatten the last two dimensions)
    x_np_flat = x_np.reshape(-1, 4)  # Reshapes [627, 2, 2] to [627, 4]
    print(f"Data shape: {x_np_flat.shape}", x_np_flat[0])  # Should show torch.Size([627, 2, 2])
    plt.figure(figsize=(15, 10))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.hist(x_np_flat[:, i], bins=30, color='skyblue', edgecolor='black')
        plt.title(f'Feature {i}')
        plt.xlabel('Value')
        plt.ylabel('Count')

    plt.tight_layout()
    plt.show()

    x_np_flat2 = x_np2.reshape(-1, 4)  # Reshapes [627, 2, 2] to [627, 4]
    print(f"Data shape: {x_np_flat2.shape}", x_np_flat2[0])  # Should show torch.Size([627, 2, 2])
    plt.figure(figsize=(15, 10))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.hist(x_np_flat2[:, i], bins=30, color='skyblue', edgecolor='black')
        plt.title(f'Feature {i}')
        plt.xlabel('Value')
        plt.ylabel('Count')

    plt.tight_layout()
    plt.show()

    # Calculate statistics on the flattened data
    mean = np.mean(x_np_flat[:, 0])
    std = np.std(x_np_flat[:, 0])
    maxx = np.max(x_np_flat[:, 0])
    minn = np.min(x_np_flat[:, 0])
    print(f"Statistics for feature 0: mean={mean:.4f}, std={std:.4f}, max={maxx:.4f}, min={minn:.4f}")


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

    output_dim = args.input_dim if args.baseline else 4
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model, field_type=args.field_type, baseline=args.baseline)
    # nn_model = MLP(input_dim=4, hidden_dim=args.hidden_dim, output_dim=output_dim, nonlinearity=args.nonlinearity)
    # model = HNN(input_dim=4, differentiable_model=nn_model, field_type=args.field_type, baseline=args.baseline)
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
    print("Shape sled vzimane", x.shape)

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


import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


# Define MLP (used as the differentiable model for the Hamiltonian)
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class MLPHNN(torch.nn.Module):
  '''Just a salt-of-the-earth MLP'''
  def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh'):
    super(MLPHNN, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
    self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=None)

    for l in [self.linear1, self.linear2, self.linear3]:
      torch.nn.init.orthogonal_(l.weight) # use a principled initialization

    self.nonlinearity = torch.nn.Tanh()

  def forward(self, x, separate_fields=False):
    h = self.nonlinearity( self.linear1(x) )
    h = self.nonlinearity( self.linear2(h) )
    return self.linear3(h)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden1_size)
        self.sigmoid = nn.Sigmoid()
        self.layer2 = nn.Linear(hidden1_size, hidden2_size)
        self.layer3 = nn.Linear(hidden2_size, output_size)

    def forward(self, x):
        x = self.sigmoid(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        x = self.layer3(x)  # No sigmoid on output (common for regression with MSELoss)
        return x
class HNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128):
        super(HNN, self).__init__()
        self.differentiable_model = MLPHNN(input_dim, hidden_dim, 1)  # Output is scalar Hamiltonian
        self.M = torch.eye(input_dim)  # Canonical coordinates matrix
        self.M = torch.cat([self.M[input_dim // 2:], -self.M[:input_dim // 2]])  # [0, 1; -1, 0] for (q̇, ṗ)

    def forward(self, x):
        """Compute Hamiltonian H(p, q)."""

        return self.differentiable_model(x)

    def time_derivative(self, x):
        """Compute time derivatives (q̇, ṗ) using Hamiltonian gradients."""
        x = x.requires_grad_(True)
        H = self.forward(x)  # Shape: [batch_size, 1]
        dH = torch.autograd.grad(H.sum(), x, create_graph=True, retain_graph=True)[0]  # Fix: Added retain_graph=True
        derivatives = dH @ self.M.t()  # [∂H/∂p, -∂H/∂q]
        return derivatives  # Returns [q̇, ṗ]
# Load and preprocess your dataset

x, dxdt, test_x, test_dxdt = getMyDataAvg()
print(x[0], x.shape)
#inputs = torch.tensor(x, dtype=torch.float32)  # [p, q]
#targets = torch.tensor(dxdt, dtype=torch.float32)  # [q̇, ṗ] (reordered to match [∂H/∂p, -∂H/∂q])

# Normalize data (optional, recommended for stability)
mean = x.mean(dim=0)
std = x.std(dim=0) + 1e-6  # Avoid division by zero
inputs = (x - mean) / std
targets = (dxdt - mean) / std  # Scale derivatives accordingly
inputsScaled, mins1, maxs1 = scale_columns_neg1_pos1(x)
targetsScaled, mins2, maxs2 = scale_columns_neg1_pos1(dxdt)
print("Sled scale", x[0], dxdt[0], inputs[0], targets[0])
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
for i in range(2):
    for j in range(2):
        axs[i, j].hist(inputsScaled[:, i, j].detach().numpy(), bins=30)
        axs[i, j].set_title(f'Column [:, {i}, {j}]')
plt.tight_layout()
#plt.show()
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
for i in range(2):
    for j in range(2):
        axs[i, j].hist(targetsScaled[:, i, j].detach().numpy(), bins=30)
        axs[i, j].set_title(f'Column [:, {i}, {j}]')
plt.tight_layout()
#plt.show()
#mean = test_x.mean(dim=0)
#std = test_x.std(dim=0) + 1e-6  # Avoid division by zero
#test_x = (test_x - mean) / std
#test_dxdt = test_dxdt / std  # Scale derivatives accordingly
test_x, mi, ma = scale_columns_neg1_pos1(test_x)
test_dxdt, mii, maa = scale_columns_neg1_pos1(test_dxdt)

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
for i in range(2):
    for j in range(2):
        axs[i, j].hist(test_x[:, i, j].detach().numpy(), bins=30)
        axs[i, j].set_title(f'Column [:, {i}, {j}]')
plt.tight_layout()
#plt.show()
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
for i in range(2):
    for j in range(2):
        axs[i, j].hist(test_dxdt[:, i, j].detach().numpy(), bins=30)
        axs[i, j].set_title(f'Column [:, {i}, {j}]')
plt.tight_layout()
#plt.show()
# Create DataLoader
dataset = TensorDataset(inputsScaled, targetsScaled)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize model and optimizer
model = HNN(input_dim=2, hidden_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

#model = NeuralNetwork(input_size=2, hidden1_size=32, hidden2_size=32, output_size=2)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
criterion = nn.MSELoss()

########################################################################
'''epochs = 5000
train_losses = []
test_losses = []

for epoch in range(epochs):
    # Training mode
    model.train()
    optimizer.zero_grad()
    train_outputs = model(inputsScaled)
    train_loss = criterion(train_outputs, targetsScaled)
    train_loss.backward(retain_graph=True)  # No retain_graph needed unless required
    optimizer.step()
    train_losses.append(train_loss.item())

    model.eval()
    with torch.no_grad():
        test_outputs = model(test_x)
        test_loss = criterion(test_outputs, test_dxdt)
        test_losses.append(test_loss.item())

    # Test mode (optional: move to separate loop)

    # Print progress every 1000 epochs
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Train Loss: {train_loss.item():.6f}, Test Loss: {test_loss.item():.6f}')

# Plotting the losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', color='#1f77b4')  # Blue for train
plt.plot(test_losses, label='Test Loss', color='#ff7f0e')  # Orange for test
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Test Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Final predictions (optional)
model.eval()
with torch.no_grad():
    print("\nFinal Test Predictions:")
    print(model(test_x))'''
#######################################################################3
# Training loop
num_epochs = 1500
train_losses = []
test_losses = []
best_test_loss = float('inf')
best_model = None

for epoch in range(num_epochs):
    # --- Training Phase ---
    model.train()
    total_train_loss = 0
    for batch_inputs, batch_targets in dataloader:
        optimizer.zero_grad()
        pred_derivatives = model.time_derivative(batch_inputs)
        loss = criterion(pred_derivatives, batch_targets)
        loss.backward(retain_graph=True)  # Keep if needed for your use case
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(dataloader)
    train_losses.append(avg_train_loss)

    # --- Evaluation Phase ---
    model.eval()

    test_pred = model.time_derivative(test_x)
    test_loss = criterion(test_pred, test_dxdt)
    test_losses.append(test_loss.item())

        # Save best model
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_model = model.state_dict()

    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}")

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', color='blue', alpha=0.7)
plt.plot(test_losses, label='Test Loss', color='red', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs. Test Loss')
plt.legend()
plt.grid(True)
plt.show()

# Final evaluation with best model
if best_model is not None:
    model.load_state_dict(best_model)
model.eval()

final_test_pred = model.time_derivative(test_x)
final_test_loss = criterion(final_test_pred, test_dxdt)
print(f"Best Test Loss: {best_test_loss:.4f}")
print(f"Final Test Loss: {final_test_loss:.4f}")

torch_tensor = torch.tensor([248.0, 130.0, 59.0 , 47.0], requires_grad=True)
torch_tensor = torch_tensor.reshape(2, 2)
#torch_tensor = (torch_tensor -mean) / std

torch_tensor = 2 * (torch_tensor - mins1) / (maxs1 - mins1) - 1
print("Posledno", torch_tensor)
dxdt_hat = model.time_derivative(torch_tensor)
print(dxdt_hat)
dxdt_hat = dxdt_hat * std
dxdt_hat = dxdt_hat + mean
print(dxdt_hat)
val1, val2 = dxdt_hat[0]
print(val1, val2)

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

    a, ra = xy_to_hamiltonian_qp(248, 130, 9, 7)  # , xy_to_hamiltonian_qp(100, 100, 0.7, 0.7)
    print("AAAA", a, ra)
    b = hamiltonian_qp_to_xy(a[0], a[1], ra[0], ra[1])
    print(b)
    torch_tensor = torch.tensor(a, requires_grad=True)
    torch_tensor = torch_tensor.reshape(1, 2)
    print("Posledno", torch_tensor)
    dxdt_hat = globalModel.time_derivative(torch_tensor)
    print(dxdt_hat)
    val1, val2 = dxdt_hat[0]
    print(val1, val2)
    bdx = hamiltonian_qp_to_xy(val1.item(), val2.item(), ra[0], ra[1])
    print(bdx)

    dxdt_hat2 = globalModel.time_derivative(dxdt_hat)

    a, ra2 = xy_to_hamiltonian_qp(bdx[0], bdx[1], bdx[2], bdx[3])  # , xy_to_hamiltonian_qp(100, 100, 0.7, 0.7)
    torch_tensor = torch.tensor(a, requires_grad=True)
    torch_tensor = torch_tensor.reshape(1, 2)
    dxdt_hat = globalModel.time_derivative(torch_tensor)
    val1, val2 = dxdt_hat[0]
    bdx = hamiltonian_qp_to_xy(val1.item(), val2.item(), ra2[0], ra2[1])
    print("BDX", bdx)

    dxdt_hat3 = globalModel.time_derivative(dxdt_hat2)
    dxdt_hat4 = globalModel.time_derivative(dxdt_hat3)
    dxdt_hat5 = globalModel.time_derivative(dxdt_hat4)
    dxdt_hat6 = globalModel.time_derivative(dxdt_hat5)

    print("##################33")
    val1, val2 = dxdt_hat2[0]
    print(val1, val2)
    bdx = hamiltonian_qp_to_xy(val1.item(), val2.item(), ra[0], ra[1])
    print(bdx)
    print("##################33")
    val1, val2 = dxdt_hat3[0]
    print(val1, val2)
    bdx = hamiltonian_qp_to_xy(val1.item(), val2.item(), ra[0], ra[1])
    print(bdx)
    print("##################33")
    val1, val2 = dxdt_hat4[0]
    print(val1, val2)
    bdx = hamiltonian_qp_to_xy(val1.item(), val2.item(), ra[0], ra[1])
    print(bdx)
    print("##################33")
    val1, val2 = dxdt_hat5[0]
    print(val1, val2)
    bdx = hamiltonian_qp_to_xy(val1.item(), val2.item(), ra[0], ra[1])
    print(bdx)
    print("##################33")
    val1, val2 = dxdt_hat6[0]
    print(val1, val2)
    bdx = hamiltonian_qp_to_xy(val1.item(), val2.item(), ra[0], ra[1])
    print(bdx)

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