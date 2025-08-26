# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski
# da razbera kak da model trajectory of projectile with hsmiltonian!!!!!!!!!!!!!
import torch, argparse
import numpy as np

import torch
from pathlib import Path
import os, sys

from torch.optim.lr_scheduler import StepLR
import copy
from scipy import stats
import pandas as pd
from numpy.ma.core import append
import seaborn as sns
from sympy import andre
from sympy.tensor import tensor
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
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

'''def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=2*2, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=600, type=int, help='batch_size')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=120, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='cleanPerf', type=str, help='only one option right now')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--field_type', default='solenoidal', type=str, help='type of vector field to learn')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.set_defaults(feature=True)
    return parser.parse_args()'''
def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=2*2, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=800, type=int, help='hidden dimension of mlp')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=600, type=int, help='batch_size')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=1200, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='cleanPerf', type=str, help='only one option right now')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--field_type', default='solenoidal', type=str, help='type of vector field to learn')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.set_defaults(feature=True)
    return parser.parse_args()

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
def parse_custom_array(s):
    # Remove np.float32() wrappers while keeping the numeric values
    cleaned = re.sub(r'np\.float32\(([^)]+)\)', r'\1', s)
    # Convert to actual list
    data = ast.literal_eval(cleaned)
    # Convert numbers to float32 if neededparse_custom_array
    numpy_array = np.array([[x[0]] + [np.float32(y) for y in x[1:]] for x in data])
    return numpy_array
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
        #if Path(mask) != Path(
                #f'C:/Users/Alexs/PyCharmMiscProject/pythonProject7/object_tracking/object_tracking/samMooooolq/segmentAnything/videos16/mask1'):
            #continue
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
            if rowindex < 8:
                prevrow.append(row)
                rowindex += 1
                continue
            i = 0
            j = 0
            indexes = []
            for itein in row:
                indexes.append(itein[0])
            print("Row Index", row, indexes)
            countsListPrevPrev = np.zeros(9)
            countsListPrev = np.zeros(9)
            countsListCurrent = np.zeros(9)
            sumsListPrevPrev = np.array([np.zeros(4) for _ in range(9)])
            sumsListPrev = np.array([np.zeros(4) for _ in range(9)])
            sumsListCurrent = np.array([np.zeros(4) for _ in range(9)])
            for index, prevRow in enumerate(prevrow):
                if index < 3:
                    for item in prevRow:
                        if item[0] in indexes:
                            # print(int(item[0]) - 1)
                            sumsListPrevPrev[int(item[0]) - 1] += item[1:]  # Add the slice (must match shape (4,))
                            countsListPrevPrev[int(item[0]) - 1] += 1
                if index >= 3 and index < 6:
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
            sumsListPrevPrev = np.divide(sumsListPrevPrev, countsListPrevPrev[:, np.newaxis],
                                     out=np.full_like(sumsListPrevPrev, np.nan),
                                     where=countsListPrevPrev[:, np.newaxis] != 0)
            filled_arrPrevPrev = np.nan_to_num(sumsListPrevPrev, nan=0, posinf=1e10, neginf=-1e10)
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
                print("In setup data")
                index = int(indexes[ind])
                print([filled_arrCurrent[index - 1].tolist()[2], filled_arrCurrent[index - 1].tolist()[3]])
                print([(filled_arrCurrent[index - 1].tolist()[0] - filled_arrPrev[index - 1].tolist()[0]) / 1,
                       (filled_arrCurrent[index - 1].tolist()[1] - filled_arrPrev[index - 1].tolist()[1]) / 1])
                print([ ((filled_arrCurrent[index - 1].tolist()[0] - filled_arrPrev[index - 1].tolist()[0]) - (filled_arrPrev[index - 1].tolist()[0] - filled_arrPrevPrev[index - 1].tolist()[0])),
                       (filled_arrCurrent[index - 1].tolist()[1] - filled_arrPrev[index - 1].tolist()[1]) - (filled_arrPrev[index - 1].tolist()[1] - filled_arrPrevPrev[index - 1].tolist()[1])])
                print(filled_arrPrev[index - 1].tolist())
                print(filled_arrCurrent[index - 1].tolist())
                print([filled_arrCurrent[index - 1].tolist()[0], filled_arrCurrent[index - 1].tolist()[1],
                       (filled_arrCurrent[index - 1].tolist()[0] - filled_arrPrev[index - 1].tolist()[0]) / 1,
                       (filled_arrCurrent[index - 1].tolist()[1] - filled_arrPrev[index - 1].tolist()[1]) / 1])
                print([ (filled_arrCurrent[index - 1].tolist()[0] - filled_arrPrev[index - 1].tolist()[0]) / 1,
                       (filled_arrCurrent[index - 1].tolist()[1] - filled_arrPrev[index - 1].tolist()[1]) / 1,
                        filled_arrCurrent[index - 1].tolist()[2], filled_arrCurrent[index - 1].tolist()[3]])

                dataset['x'].append([[filled_arrCurrent[index - 1].tolist()[0], filled_arrCurrent[index - 1].tolist()[1]],
                                     [(filled_arrCurrent[index - 1].tolist()[0] - filled_arrPrev[index - 1].tolist()[0]) / 1,
                                      (filled_arrCurrent[index - 1].tolist()[1] - filled_arrPrev[index - 1].tolist()[1]) / 1]])
                dataset['dx'].append([[(filled_arrCurrent[index - 1].tolist()[0] - filled_arrPrev[index - 1].tolist()[0]) / 1,
                                       (filled_arrCurrent[index - 1].tolist()[1] - filled_arrPrev[index - 1].tolist()[1]) / 1],
                                      [((filled_arrCurrent[index - 1].tolist()[0] - filled_arrPrev[index - 1].tolist()[
                                          0]) - (filled_arrPrev[index - 1].tolist()[0] -
                                                 filled_arrPrevPrev[index - 1].tolist()[0])),
                                       (filled_arrCurrent[index - 1].tolist()[1] - filled_arrPrev[index - 1].tolist()[
                                           1]) - (filled_arrPrev[index - 1].tolist()[1] -
                                                  filled_arrPrevPrev[index - 1].tolist()[1])]])

            # print("prevrow", prevrow[0][0], prevrow[1][0], prevrow[2][0], prevrow[4][0])
            prevrow.pop(0)
            prevrow.append(row)
            # print("prevrow", prevrow[0][0], prevrow[1][0], prevrow[2][0], prevrow[4][0])


        allData.append(dataset)
        print("Real data1", dataset['x'][0], dataset['dx'][0])
        for data in allData:

            for key in data:
                print("The key", key)
                std_population = np.std(data[key], axis=0)
                mean_population = np.mean(data[key], axis=0)
                print(std_population)
                print(mean_population)
                if key == 'dx':
                    std_population = std_population * [[1, 1], [0.2, 0.2]]
                for pair_list in data[key]:
                    # Check if the pair_list has at least two elements (assuming it's always pairs)
                    if len(pair_list) >= 2:

                        # Iterate through each element in the second list of the pair
                        '''for j in range(2):
                            for i in range(len(pair_list[j])):
                                if (pair_list[j][i] > mean_population[j][i] + 0.02 * std_population[j][i]):
                                    pair_list[j][i] = mean_population[j][i] + 0.02 * std_population[j][i]
                                if pair_list[j][i] < mean_population[j][i] - 0.02 * std_population[j][i]:
                                    pair_list[j][i] = mean_population[j][i] - 0.02 * std_population[j][i]'''
                        if key == 'dx' and pair_list[1][0] < -150:
                            pair_list[1][0] = -150
                        if key == 'dx' and pair_list[1][0] > 150:
                            pair_list[1][0] = 150
                        if key == 'dx' and pair_list[1][1] < -150:
                            pair_list[1][1] = -150
                        if key == 'dx' and pair_list[1][1] > 150:
                            pair_list[1][1] = 150

                        if key == 'dx' and pair_list[0][0] < -60:
                            pair_list[0][0] = -60
                        if key == 'dx' and pair_list[0][0] > 80:
                            pair_list[0][0] = 80
                        if key == 'dx' and pair_list[0][1] < -50:
                            pair_list[0][1] = -50
                        if key == 'dx' and pair_list[0][1] > 60:
                            pair_list[0][1] = 60

                        if key == 'x' and pair_list[1][0] < -60:
                            pair_list[1][0] = -60
                        if key == 'x' and pair_list[1][0] > 80:
                            pair_list[1][0] = 80
                        if key == 'x' and pair_list[1][1] < -50:
                            pair_list[1][1] = -50
                        if key == 'x' and pair_list[1][1] > 60:
                            pair_list[1][1] = 60


        print("Real data2", dataset['x'][0], dataset['dx'][0])

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

        print("######Proverka#####", len(split_dict['x']), len(test_dataset['x']), len(split_dict['dx']), len(test_dataset['dx']),
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
    '''plt.figure(figsize=(15, 10))
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
    plt.show()'''

    '''correlation = np.corrcoef(x_np_flat[:, 0], x_np_flat[:, 2])[0, 1]
    print(f"Pearson Correlation (r): {correlation:.3f}")
    correlation = np.corrcoef(x_np_flat[:, 1], x_np_flat[:, 3])[0, 1]
    print(f"Pearson Correlation (r): {correlation:.3f}")
    nsc = np.arange(len(x_np_flat)/ 10)
    xsc = x_np_flat[::10, 0]  # Column 0
    ysc = x_np_flat[::10, 2]  # Column 2

    plt.figure(figsize=(10, 6))
    plt.plot(nsc, xsc, color='blue', label='Column 0')
    plt.plot(nsc, ysc, color='red', label='Column 2')
    plt.xlabel("Index (or your X-axis label)")
    plt.ylabel("Value")
    plt.title("Trend Comparison: Column 0 (Blue) vs Column 2 (Red)")
    plt.legend()
    plt.grid(True)
    plt.show()

    nsc = np.arange(len(x_np_flat) / 10)
    xsc = x_np_flat[::10, 1]  # Column 0
    ysc = x_np_flat[::10, 3]  # Column 2

    plt.figure(figsize=(10, 6))
    plt.plot(nsc, xsc, color='blue', label='Column 0')
    plt.plot(nsc, ysc, color='red', label='Column 2')
    plt.xlabel("Index (or your X-axis label)")
    plt.ylabel("Value")
    plt.title("Trend Comparison: Column 0 (Blue) vs Column 2 (Red)")
    plt.legend()
    plt.grid(True)
    plt.show()
    # Calculate statistics on the flattened data
    mean = np.mean(x_np_flat[:, 0])
    std = np.std(x_np_flat[:, 0])
    maxx = np.max(x_np_flat[:, 0])
    minn = np.min(x_np_flat[:, 0])
    print(f"Statistics for feature 0: mean={mean:.4f}, std={std:.4f}, max={maxx:.4f}, min={minn:.4f}")'''


    #return x, dxdt, test_x, test_dxdt
    #x, dxdt, test_x, test_dxdt = getMyDataAvg()
    #print(x[0], x.shape)
    # inputs = torch.tensor(x, dtype=torch.float32)  # [p, q]
    # targets = torch.tensor(dxdt, dtype=torch.float32)  # [q̇, ṗ] (reordered to match [∂H/∂p, -∂H/∂q])

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
    plt.show()
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for i in range(2):
        for j in range(2):
            axs[i, j].hist(targetsScaled[:, i, j].detach().numpy(), bins=30)
            axs[i, j].set_title(f'Column [:, {i}, {j}]')
    plt.tight_layout()
    plt.show()
    # mean = test_x.mean(dim=0)
    # std = test_x.std(dim=0) + 1e-6  # Avoid division by zero
    # test_x = (test_x - mean) / std
    # test_dxdt = test_dxdt / std  # Scale derivatives accordingly
    test_x, mi, ma = scale_columns_neg1_pos1(test_x)
    test_dxdt, mii, maa = scale_columns_neg1_pos1(test_dxdt)

    '''fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for i in range(2):
        for j in range(2):
            axs[i, j].hist(test_x[:, i, j].detach().numpy(), bins=30)
            axs[i, j].set_title(f'Column [:, {i}, {j}]')
    plt.tight_layout()
    plt.show()
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for i in range(2):
        for j in range(2):
            axs[i, j].hist(test_dxdt[:, i, j].detach().numpy(), bins=30)
            axs[i, j].set_title(f'Column [:, {i}, {j}]')
    plt.tight_layout()
    plt.show()'''
    # Create DataLoader
    inputsScaled = inputsScaled.reshape(-1, 4)
    targetsScaled = targetsScaled.reshape(-1, 4)
    test_x = test_x.reshape(-1, 4)
    test_dxdt = test_dxdt.reshape(-1, 4)
    print(inputsScaled.shape, targetsScaled.shape, test_x.shape, test_dxdt.shape)
    return inputsScaled, targetsScaled, test_x, test_dxdt, mins1, maxs1, mins2, maxs2


def train(args):
  # set random seed
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  args.verbose = True

  # init model and optimizer
  if args.verbose:
    print("Training baseline model:" if args.baseline else "Training HNN model:")

  output_dim = args.input_dim if args.baseline else 2
  nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
  model = HNN(args.input_dim, differentiable_model=nn_model,
            field_type=args.field_type, baseline=args.baseline)

  optim = torch.optim.Adam(model.parameters(), args.learn_rate)#, weight_decay=1e-4
  #scheduler = StepLR(optim, step_size=200, gamma=0.1)


  # arrange data
  '''data = get_dataset(args.name, args.save_dir, verbose=True)
  x = torch.tensor( data['coords'], requires_grad=True, dtype=torch.float32)
  test_x = torch.tensor( data['test_coords'], requires_grad=True, dtype=torch.float32)
  dxdt = torch.Tensor(data['dcoords'])
  test_dxdt = torch.Tensor(data['test_dcoords'])'''

  x, dxdt, test_x, test_dxdt, min1, max1, min2, max2 = getMyDataAvg()

  # vanilla train loop
  stats = {'train_loss': [], 'test_loss': []}
  for step in range(args.total_steps+1):

    # train step
    ixs = torch.randperm(x.shape[0])[:args.batch_size]
    dxdt_hat = model.time_derivative(x[ixs])
    loss = L2_loss(dxdt[ixs], dxdt_hat)
    loss.backward(retain_graph=True)
    grad = torch.cat([p.grad.flatten() for p in model.parameters()]).clone()
    optim.step() ; optim.zero_grad()
    #scheduler.step()

    # run test data
    test_ixs = torch.randperm(test_x.shape[0])[:args.batch_size]
    test_dxdt_hat = model.time_derivative(test_x[test_ixs])
    test_loss = L2_loss(test_dxdt[test_ixs], test_dxdt_hat)

    # logging
    stats['train_loss'].append(loss.item())
    stats['test_loss'].append(test_loss.item())
    if args.verbose and step % args.print_every == 0:
      print("step {}, train_loss {:.4e}, test_loss {:.4e}, grad norm {:.4e}, grad std {:.4e}"
          .format(step, loss.item(), test_loss.item(), grad@grad, grad.std()))

  plt.figure(figsize=(10, 6))
  plt.plot(stats['train_loss'], label='Training loss')
  plt.plot(stats['test_loss'], label='Test loss')
  plt.xlabel('Training steps')
  plt.ylabel('Loss')
  plt.title('Training and Test Loss over Time')
  plt.legend()
  plt.grid(True)

  # Use logarithmic scale if your losses span several orders of magnitude
  plt.yscale('log')  # Optional - useful for wide-ranging loss values

  plt.show()
  train_dxdt_hat = model.time_derivative(x)
  train_dist = (dxdt - train_dxdt_hat)**2
  test_dxdt_hat = model.time_derivative(test_x)
  test_dist = (test_dxdt - test_dxdt_hat)**2
  print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
    .format(train_dist.mean().item(), train_dist.std().item()/np.sqrt(train_dist.shape[0]),
            test_dist.mean().item(), test_dist.std().item()/np.sqrt(test_dist.shape[0])))

  outputs = []
  torch_tensororiginal = torch.tensor([82.6668, 180.6123, 12.4338, -5.1671])
  outputs.append(torch_tensororiginal.clone())
  #torch_tensororiginal = torch.tensor([248.0, 200.0, 5.0, 4.0])

  torch_tensororiginal = torch_tensororiginal.reshape(1, 4)
  min1 = min1.reshape(1, 4)
  max1 = max1.reshape(1, 4)
  min2 = min2.reshape(1, 4)
  max2 = max2.reshape(1, 4)


  print(min1, max1)
  print(min2, max2)
  print("Original in coordinates", torch_tensororiginal)
  torch_tensor = 2 * (torch_tensororiginal - min1) / (max1 - min1) - 1
  print("Normalized original", torch_tensor)
  dxdt_hat = model.time_derivative(torch_tensor)
  print("Normalized output", dxdt_hat)
  dxdt_hat =  min2 + (dxdt_hat + 1) * (max2 - min2) / 2
  print("###############################")
  print("Output in coordinates", dxdt_hat)#
  print("###############################")
  newcoord = torch_tensororiginal

  for i in  range(4):
      print(newcoord)
      oldx = newcoord[0][0].clone()
      oldy = newcoord[0][1].clone()


      ############# Tova tuka s 1/2ta e Taylor expansion
      newcoord[0][0] = newcoord[0][0] + dxdt_hat[0][0] + dxdt_hat[0][2] / 2
      newcoord[0][1] = newcoord[0][1] + dxdt_hat[0][1] + dxdt_hat[0][3] / 2
      print(newcoord, oldx, oldy)
      newcoord[0][2] = newcoord[0][2] + dxdt_hat[0][2]
      newcoord[0][3] = newcoord[0][3] + dxdt_hat[0][3]
      print(newcoord)
      outputs.append(newcoord.clone())

      print("Original in coordinates", newcoord)
      torch_tensor = 2 * (newcoord - min1) / (max1 - min1) - 1
      print("Normalized original", torch_tensor)
      dxdt_hat = model.time_derivative(torch_tensor)
      print("Normalized output", dxdt_hat)
      dxdt_hat =  min2 + (dxdt_hat + 1) * (max2 - min2) / 2
      print("###############################")
      print("Output in coordinates", dxdt_hat)#
      print("###############################")

  print(outputs)




  return model, stats

if __name__ == "__main__":
    args = get_args()
    args.baseline = False
    model, stats = train(args)

    # save model
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    label = 'baseline' if args.baseline else 'hnn'
    model_path = '{}/{}-orbits2-{}.tar'.format(args.save_dir, args.name, label)
    torch.save(model.state_dict(), model_path)

    # save stats
    stats_path = '{}/{}-orbits-{}.pkl'.format(args.save_dir, args.name, label)
    to_pickle(stats, stats_path)