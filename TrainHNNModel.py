from pathlib import Path
import os, sys

# Get current directory and parent directory paths
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add parent directory to Python path for module imports
sys.path.append(PARENT_DIR)

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import re
import ast
import torch, argparse
import numpy as np
import os, sys

# Re-define directory paths (duplicate code - may be intentional for module imports)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

# Import custom modules
from nn_models import MLP
from hnn import HNN
from utils import L2_loss, to_pickle, from_pickle
from Config import paths

# Get main folder path from configuration
main_folder = paths['mainfolder']


def scale_columns_neg1_pos1(tensor):
    """
    Scale each column of a tensor to the range [-1, 1].

    Args:
        tensor: Input tensor to scale

    Returns:
        scaled_tensor: Tensor scaled to [-1, 1] range
        col_mins: Minimum values for each column (for inverse scaling)
        col_maxs: Maximum values for each column (for inverse scaling)
    """
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
    """
    Parse command line arguments for model configuration.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=2 * 2, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=400, type=int, help='hidden dimension of mlp')
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
    """
    Find all mask subdirectories within video directories.

    Args:
        directory_path: Root directory containing video folders

    Returns:
        List of paths to mask subdirectories
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"'{directory_path}' is not a valid directory")

    # Pattern to match video directories (videos1, videos2, ..., videos29)
    video_pattern = re.compile(r'^videos([1-2]?[0-9])$')
    # Pattern to match mask directories (mask0, mask1, mask2, ...)
    mask_pattern = re.compile(r'^mask\d+$')

    mask_subdirs = []

    # Iterate through all directories in the main folder
    for video_dir in os.listdir(directory_path):
        video_path = os.path.join(directory_path, video_dir)
        # Check if it's a video directory matching the pattern
        if os.path.isdir(video_path) and video_pattern.match(video_dir):
            # Look for mask subdirectories within the video directory
            for subdir in os.listdir(video_path):
                subdir_path = os.path.join(video_path, subdir)
                if os.path.isdir(subdir_path) and mask_pattern.match(subdir):
                    mask_subdirs.append(subdir_path)

    return sorted(mask_subdirs)


def parse_custom_array(s):
    """
    Parse a string representation of a numpy array with float32 values.

    Args:
        s: String representation of array

    Returns:
        numpy_array: Parsed numpy array
    """
    # Remove np.float32() wrappers while keeping the numeric values
    cleaned = re.sub(r'np\.float32\(([^)]+)\)', r'\1', s)
    # Convert to actual list using ast.literal_eval
    data = ast.literal_eval(cleaned)
    # Convert numbers to float32 if needed
    numpy_array = np.array([[x[0]] + [np.float32(y) for y in x[1:]] for x in data])
    return numpy_array


def getMyDataAvg():
    """
    Main data loading and preprocessing function.
    Loads tracking data from CSV files, processes it, and prepares for training.

    Returns:
        inputsScaled: Scaled training inputs
        targetsScaled: Scaled training targets
        test_x: Scaled test inputs
        test_dxdt: Scaled test targets
        mins1: Min values for input scaling
        maxs1: Max values for input scaling
        mins2: Min values for target scaling
        maxs2: Max values for target scaling
    """
    # Initialize tensors for storing data
    x = torch.tensor([])
    dxdt = torch.tensor([])
    test_x = torch.tensor([])
    test_dxdt = torch.tensor([])
    allData = []
    i = 0

    # Iterate through all mask directories
    for mask in get_mask_subdirs_os2(main_folder):
        # Skip specific mask directory (hardcoded exclusion)
        if Path(mask) == Path(main_folder + f'/videos13/mask3'):
            continue

        # Read training data CSV file
        df = pd.read_csv(mask + '/trainData.csv')

        rowindex = 0
        prevrow = []  # Store previous rows for velocity calculation
        dataset = {}
        dataset['x'] = []
        dataset['dx'] = []

        # Process each row in the CSV
        for row in df['cap']:
            row = parse_custom_array(row)
            if row.size == 0:
                continue

            # Collect first 8 rows for initialization
            if rowindex < 8:
                prevrow.append(row)
                rowindex += 1
                continue

            # Extract indexes from current row
            indexes = []
            for itein in row:
                indexes.append(itein[0])

            # Initialize arrays for averaging
            countsListPrevPrev = np.zeros(9)
            countsListPrev = np.zeros(9)
            countsListCurrent = np.zeros(9)
            sumsListPrevPrev = np.array([np.zeros(4) for _ in range(9)])
            sumsListPrev = np.array([np.zeros(4) for _ in range(9)])
            sumsListCurrent = np.array([np.zeros(4) for _ in range(9)])

            # Process previous rows (sliding window of 8 rows)
            for index, prevRow in enumerate(prevrow):
                if index < 3:
                    for item in prevRow:
                        if item[0] in indexes:
                            sumsListPrevPrev[int(item[0]) - 1] += item[1:]  # Add the slice (must match shape (4,))
                            countsListPrevPrev[int(item[0]) - 1] += 1
                if index >= 3 and index < 6:
                    for item in prevRow:
                        if item[0] in indexes:
                            sumsListPrev[int(item[0]) - 1] += item[1:]  # Add the slice (must match shape (4,))
                            countsListPrev[int(item[0]) - 1] += 1
                else:
                    for item in prevRow:
                        if item[0] in indexes:
                            sumsListCurrent[int(item[0]) - 1] += item[1:]  # Add the slice (must match shape (4,))
                            countsListCurrent[int(item[0]) - 1] += 1

            # Add current row data
            for item in row:
                sumsListCurrent[int(item[0]) - 1] += item[1:]  # Add the slice (must match shape (4,))
                countsListCurrent[int(item[0]) - 1] += 1

            # Calculate averages for each time window, handling division by zero
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

            # Create dataset entries for each tracked object
            for ind in range(len(row)):
                index = int(indexes[ind])

                # State vector: [current_position, velocity]
                dataset['x'].append(
                    [[filled_arrCurrent[index - 1].tolist()[0], filled_arrCurrent[index - 1].tolist()[1]],
                     [(filled_arrCurrent[index - 1].tolist()[0] - filled_arrPrev[index - 1].tolist()[0]) / 1,
                      (filled_arrCurrent[index - 1].tolist()[1] - filled_arrPrev[index - 1].tolist()[1]) / 1]])

                # Derivative: [velocity, acceleration]
                dataset['dx'].append(
                    [[(filled_arrCurrent[index - 1].tolist()[0] - filled_arrPrev[index - 1].tolist()[0]) / 1,
                      (filled_arrCurrent[index - 1].tolist()[1] - filled_arrPrev[index - 1].tolist()[1]) / 1],
                     [((filled_arrCurrent[index - 1].tolist()[0] - filled_arrPrev[index - 1].tolist()[
                         0]) - (filled_arrPrev[index - 1].tolist()[0] -
                                filled_arrPrevPrev[index - 1].tolist()[0])),
                      (filled_arrCurrent[index - 1].tolist()[1] - filled_arrPrev[index - 1].tolist()[
                          1]) - (filled_arrPrev[index - 1].tolist()[1] -
                                 filled_arrPrevPrev[index - 1].tolist()[1])]])

            # Update sliding window: remove oldest, add current
            prevrow.pop(0)
            prevrow.append(row)

        allData.append(dataset)

        # Apply clipping to handle outliers in the data
        for data in allData:
            for key in data:
                for pair_list in data[key]:
                    # Check if the pair_list has at least two elements (assuming it's always pairs)
                    if len(pair_list) >= 2:
                        # Clip acceleration values
                        if key == 'dx' and pair_list[1][0] < -150:
                            pair_list[1][0] = -150
                        if key == 'dx' and pair_list[1][0] > 150:
                            pair_list[1][0] = 150
                        if key == 'dx' and pair_list[1][1] < -150:
                            pair_list[1][1] = -150
                        if key == 'dx' and pair_list[1][1] > 150:
                            pair_list[1][1] = 150

                        # Clip velocity values
                        if key == 'dx' and pair_list[0][0] < -60:
                            pair_list[0][0] = -60
                        if key == 'dx' and pair_list[0][0] > 80:
                            pair_list[0][0] = 80
                        if key == 'dx' and pair_list[0][1] < -50:
                            pair_list[0][1] = -50
                        if key == 'dx' and pair_list[0][1] > 60:
                            pair_list[0][1] = 60

                        # Clip position values (from velocity component of state)
                        if key == 'x' and pair_list[1][0] < -60:
                            pair_list[1][0] = -60
                        if key == 'x' and pair_list[1][0] > 80:
                            pair_list[1][0] = 80
                        if key == 'x' and pair_list[1][1] < -50:
                            pair_list[1][1] = -50
                        if key == 'x' and pair_list[1][1] > 60:
                            pair_list[1][1] = 60

        # Split data into training and test sets
        split_dict, test_dataset = split_data_with_shuffle(dataset)

        # Append to overall tensors
        x = torch.cat((x, torch.tensor(split_dict['x'], requires_grad=True, dtype=torch.float32)), dim=0)
        dxdt = torch.cat((dxdt, torch.tensor(split_dict['dx'], requires_grad=True, dtype=torch.float32)), dim=0)
        test_x = torch.cat((test_x, torch.tensor(test_dataset['x'], requires_grad=True, dtype=torch.float32)), dim=0)
        test_dxdt = torch.cat((test_dxdt, torch.tensor(test_dataset['dx'], requires_grad=True, dtype=torch.float32)),
                              dim=0)

    # Convert to numpy for visualization
    x_np = x.detach().numpy()
    # Reshape the data for plotting (flatten the last two dimensions)
    x_np_flat = x_np.reshape(-1, 4)  # Reshapes [627, 2, 2] to [627, 4]

    # Normalize data using z-score normalization (commented out - not used)
    # mean = x.mean(dim=0)
    # std = x.std(dim=0) + 1e-6  # Avoid division by zero
    # inputs = (x - mean) / std
    # targets = (dxdt - mean) / std  # Scale derivatives accordingly

    # Scale data to [-1, 1] range
    inputsScaled, mins1, maxs1 = scale_columns_neg1_pos1(x)
    targetsScaled, mins2, maxs2 = scale_columns_neg1_pos1(dxdt)

    # Visualize input data distribution
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for i in range(2):
        for j in range(2):
            axs[i, j].hist(inputsScaled[:, i, j].detach().numpy(), bins=30)
            axs[i, j].set_title(f'Column [:, {i}, {j}]')
    plt.tight_layout()
    plt.show()

    # Visualize target data distribution
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for i in range(2):
        for j in range(2):
            axs[i, j].hist(targetsScaled[:, i, j].detach().numpy(), bins=30)
            axs[i, j].set_title(f'Column [:, {i}, {j}]')
    plt.tight_layout()
    plt.show()

    # Scale test data
    test_x, mi, ma = scale_columns_neg1_pos1(test_x)
    test_dxdt, mii, maa = scale_columns_neg1_pos1(test_dxdt)

    # Reshape data from [batch, 2, 2] to [batch, 4] for neural network input
    inputsScaled = inputsScaled.reshape(-1, 4)
    targetsScaled = targetsScaled.reshape(-1, 4)
    test_x = test_x.reshape(-1, 4)
    test_dxdt = test_dxdt.reshape(-1, 4)

    return inputsScaled, targetsScaled, test_x, test_dxdt, mins1, maxs1, mins2, maxs2


def train(args):
    """
    Main training function for the HNN or baseline model.

    Args:
        args: Command line arguments

    Returns:
        model: Trained model
        stats: Training statistics
    """
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.verbose = True

    # init model and optimizer
    if args.verbose:
        print("Training baseline model:" if args.baseline else "Training HNN model:")

    # Determine output dimension: same as input for baseline, 2 for HNN (Hamiltonian)
    output_dim = args.input_dim if args.baseline else 2
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model,
                field_type=args.field_type, baseline=args.baseline)

    # Initialize optimizer
    optim = torch.optim.Adam(model.parameters(), args.learn_rate)  # weight_decay=1e-4 commented out

    # Load data
    x, dxdt, test_x, test_dxdt, min1, max1, min2, max2 = getMyDataAvg()

    # vanilla train loop
    stats = {'train_loss': [], 'test_loss': []}
    for step in range(args.total_steps + 1):
        # train step
        ixs = torch.randperm(x.shape[0])[:args.batch_size]
        dxdt_hat = model.time_derivative(x[ixs])
        loss = L2_loss(dxdt[ixs], dxdt_hat)
        loss.backward(retain_graph=True)
        grad = torch.cat([p.grad.flatten() for p in model.parameters()]).clone()
        optim.step()
        optim.zero_grad()
        # scheduler.step()  # commented out

        # run test data
        test_ixs = torch.randperm(test_x.shape[0])[:args.batch_size]
        test_dxdt_hat = model.time_derivative(test_x[test_ixs])
        test_loss = L2_loss(test_dxdt[test_ixs], test_dxdt_hat)

        # logging
        stats['train_loss'].append(loss.item())
        stats['test_loss'].append(test_loss.item())
        if args.verbose and step % args.print_every == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}, grad norm {:.4e}, grad std {:.4e}"
                  .format(step, loss.item(), test_loss.item(), grad @ grad, grad.std()))

    # Plot training and test loss
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

    # Calculate final performance metrics
    train_dxdt_hat = model.time_derivative(x)
    train_dist = (dxdt - train_dxdt_hat) ** 2
    test_dxdt_hat = model.time_derivative(test_x)
    test_dist = (test_dxdt - test_dxdt_hat) ** 2
    print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
          .format(train_dist.mean().item(), train_dist.std().item() / np.sqrt(train_dist.shape[0]),
                  test_dist.mean().item(), test_dist.std().item() / np.sqrt(test_dist.shape[0])))

    return model, stats


if __name__ == "__main__":
    # Main execution block
    args = get_args()
    args.baseline = False  # Force HNN training (not baseline)
    model, stats = train(args)

    # save model
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    label = 'baseline' if args.baseline else 'hnn'
    model_path = '{}/{}-orbits2-{}.tar'.format(args.save_dir, args.name, label)
    torch.save(model.state_dict(), model_path)

    # save stats
    stats_path = '{}/{}-orbits-{}.pkl'.format(args.save_dir, args.name, label)
    to_pickle(stats, stats_path)