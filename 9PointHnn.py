# Import necessary libraries and modules
from pathlib import Path
import os, sys

# Set up paths to allow importing from parent directory
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

# Import project configuration
from Config import paths

main_folder = paths['mainfolder']

# Import data manipulation and visualization libraries
import pandas as pd
import matplotlib.pyplot as plt
import re  # For regular expressions
import ast  # For safely evaluating strings containing Python expressions

# Re-import and re-set paths (redundant - could be removed)
import os, sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

# Import custom neural network modules
from nn_models import MLP
from hnn import HNN
from utils import L2_loss, to_pickle, from_pickle

# Import deep learning and argument parsing libraries
import torch, argparse
import numpy as np


def scale_columns_neg1_pos1(tensor):
    """
    Scale each column of a tensor to the range [-1, 1].

    Parameters:
    -----------
    tensor : torch.Tensor
        Input tensor with shape (N, M) where N is number of samples, M is number of features

    Returns:
    --------
    scaled_tensor : torch.Tensor
        Scaled tensor with same shape as input
    col_mins : torch.Tensor
        Minimum values for each column (used for inverse scaling)
    col_maxs : torch.Tensor
        Maximum values for each column (used for inverse scaling)
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
    Parse command line arguments for the neural network training.

    Returns:
    --------
    args : argparse.Namespace
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=36, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=1200, type=int, help='hidden dimension of mlp')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=600, type=int, help='batch_size')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=400, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='cleanPerf', type=str, help='only one option right now')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--field_type', default='solenoidal', type=str, help='type of vector field to learn')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.set_defaults(feature=True)
    return parser.parse_args()


def removeOutliyers(data, columns_to_clip):
    """
    Remove outliers from tensor data by clipping values based on column statistics.

    Parameters:
    -----------
    data : torch.Tensor
        Input data tensor
    columns_to_clip : list or None
        List of column indices to clip. If None, clip all columns using global statistics.

    Returns:
    --------
    clipped_data : torch.Tensor
        Data with outliers clipped
    """
    # Calculate global mean and std
    clipped_data = data.clone()

    if columns_to_clip is None:
        # Clip all columns using global statistics (original behavior)
        mean = torch.mean(data)
        std = torch.std(data)
        lower_bound = mean - std
        upper_bound = mean + std
        clipped_data = torch.clamp(data, min=lower_bound, max=upper_bound)
    else:
        # Calculate mean and std for EACH COLUMN individually
        column_means = torch.mean(data, dim=0)  # Shape: (n_features,)
        column_stds = torch.std(data, dim=0)  # Shape: (n_features,)

        # Clip only the specified columns using their own statistics
        for col in columns_to_clip:
            lower_bound = column_means[col] - 0.5 * column_stds[col]
            upper_bound = column_means[col] + 0.5 * column_stds[col]

            clipped_data[:, col] = torch.clamp(
                data[:, col],
                min=lower_bound,
                max=upper_bound
            )
    print("Original data shape:", data.shape)
    print("Clipped data shape:", clipped_data.shape)  # Should be the same
    return clipped_data


def split_data_with_shuffle(data, test_size=0.2, random_state=None):
    """
    Splits a dictionary with 'x' and 'dx' keys into train and test sets with shuffling.

    Parameters:
    -----------
    data : dict
        Dictionary with keys 'x' and 'dx' containing data and labels
    test_size : float
        Proportion of data to include in test split (default 0.2)
    random_state : int or None
        Seed for random shuffling (optional)

    Returns:
    --------
    trainData : dict
        Dictionary with training data
    testData : dict
        Dictionary with test data
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
    Recursively find all mask subdirectories within video directories.

    Parameters:
    -----------
    directory_path : str
        Path to the main directory containing video folders

    Returns:
    --------
    mask_subdirs : list
        Sorted list of paths to mask subdirectories
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"'{directory_path}' is not a valid directory")

    # Regex patterns to match video directories (e.g., 'videos1', 'videos13') and mask directories (e.g., 'mask1')
    video_pattern = re.compile(r'^videos([1-2]?[0-9])$')
    mask_pattern = re.compile(r'^mask\d+$')

    mask_subdirs = []

    # Traverse directory structure
    for video_dir in os.listdir(directory_path):
        video_path = os.path.join(directory_path, video_dir)
        if os.path.isdir(video_path) and video_pattern.match(video_dir):
            for subdir in os.listdir(video_path):
                subdir_path = os.path.join(video_path, subdir)
                if os.path.isdir(subdir_path) and mask_pattern.match(subdir):
                    mask_subdirs.append(subdir_path)

    return sorted(mask_subdirs)


def parse_custom_array(s):
    """
    Parse a string representation of a numpy array with float32 values.

    Parameters:
    -----------
    s : str
        String representation of array, e.g., "[[np.float32(1), np.float32(2)], ...]"

    Returns:
    --------
    numpy_array : np.ndarray
        Parsed numpy array
    """
    # Remove np.float32() wrappers while keeping the numeric values
    cleaned = re.sub(r'np\.float32\(([^)]+)\)', r'\1', s)
    # Convert to actual list using ast.literal_eval (safer than eval)
    data = ast.literal_eval(cleaned)
    # Convert numbers to float32 if needed
    numpy_array = np.array([[x[0]] + [np.float32(y) for y in x[1:]] for x in data])
    return numpy_array


def getMyDataAvg():
    """
    Main data loading and preprocessing function.
    Loads data from multiple mask directories, processes it, and prepares for training.

    Returns:
    --------
    inputsScaled : torch.Tensor
        Scaled training input data
    targetsScaled : torch.Tensor
        Scaled training target data (derivatives)
    test_x : torch.Tensor
        Scaled test input data
    test_dxdt : torch.Tensor
        Scaled test target data
    mins1, maxs1 : torch.Tensor
        Min and max values for input scaling (for potential inverse transformation)
    mins2, maxs2 : torch.Tensor
        Min and max values for target scaling (for potential inverse transformation)
    """
    # Display the DataFrame
    x = torch.tensor([])  # Training inputs
    dxdt = torch.tensor([])  # Training targets (derivatives)
    test_x = torch.tensor([])  # Test inputs
    test_dxdt = torch.tensor([])  # Test targets
    allData = []  # List to store all datasets
    i = 0

    # Process each mask directory
    for mask in get_mask_subdirs_os2(main_folder):
        # Skip specific mask directory (hardcoded exclusion)
        if Path(mask) == Path(main_folder + f'/videos13/mask3'):
            continue

        # Load CSV data for this mask
        df = pd.read_csv(mask + '/trainData.csv')

        rowindex = 0
        prevrow = []  # Store previous rows for calculating derivatives
        dataset = {}
        dataset['x'] = []
        dataset['dx'] = []

        # Process each row in the 'cap' column
        for row in df['cap']:
            row = parse_custom_array(row)
            if row.size == 0:
                continue

            # Build buffer of previous rows (needs 8 previous rows for calculations)
            if rowindex < 8:
                prevrow.append(row)
                rowindex += 1
                continue

            # Extract current frame indexes
            indexes = []
            for itein in row:
                indexes.append(itein[0])

            # Initialize arrays for calculating averages
            countsListPrevPrev = np.zeros(9)
            countsListPrev = np.zeros(9)
            countsListCurrent = np.zeros(9)
            sumsListPrevPrev = np.array([np.zeros(4) for _ in range(9)])
            sumsListPrev = np.array([np.zeros(4) for _ in range(9)])
            sumsListCurrent = np.array([np.zeros(4) for _ in range(9)])

            # Process previous rows to calculate moving averages
            for index, prevRow in enumerate(prevrow):
                if index < 3:
                    # First 3 previous frames (prev-prev time step)
                    for item in prevRow:
                        if item[0] in indexes:
                            sumsListPrevPrev[int(item[0]) - 1] += item[1:]  # Add the slice (must match shape (4,))
                            countsListPrevPrev[int(item[0]) - 1] += 1
                elif index >= 3 and index < 6:
                    # Next 3 previous frames (prev time step)
                    for item in prevRow:
                        if item[0] in indexes:
                            sumsListPrev[int(item[0]) - 1] += item[1:]  # Add the slice (must match shape (4,))
                            countsListPrev[int(item[0]) - 1] += 1
                else:
                    # Last 2 previous frames (current time step, minus current row)
                    for item in prevRow:
                        if item[0] in indexes:
                            sumsListCurrent[int(item[0]) - 1] += item[1:]  # Add the slice (must match shape (4,))
                            countsListCurrent[int(item[0]) - 1] += 1

            # Add current row data
            for item in row:
                sumsListCurrent[int(item[0]) - 1] += item[1:]  # Add the slice (must match shape (4,))
                countsListCurrent[int(item[0]) - 1] += 1

            # Calculate averages, handling division by zero
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

            # Construct feature vectors for each of the 9 tracked objects
            xdatap = []  # Position features
            dxdatap = []  # Velocity features (first derivative)
            xdataq = []  # Velocity as part of state
            dxdataq = []  # Acceleration features (second derivative)
            xdata = []  # Complete state vector
            dxdata = []  # Complete derivative vector

            for index in range(9):
                # Position (p) components
                xdatap.append(filled_arrCurrent[index - 1].tolist()[0])
                xdatap.append(filled_arrCurrent[index - 1].tolist()[1])

                # Velocity (q) components (difference from previous)
                xdataq.append(filled_arrCurrent[index - 1].tolist()[0] - filled_arrPrev[index - 1].tolist()[0])
                xdataq.append(filled_arrCurrent[index - 1].tolist()[1] - filled_arrPrev[index - 1].tolist()[1])

                # First derivatives (velocity)
                dxdatap.append(filled_arrCurrent[index - 1].tolist()[0] - filled_arrPrev[index - 1].tolist()[0])
                dxdatap.append(filled_arrCurrent[index - 1].tolist()[1] - filled_arrPrev[index - 1].tolist()[1])

                # Second derivatives (acceleration)
                dxdataq.append((filled_arrCurrent[index - 1].tolist()[0] - filled_arrPrev[index - 1].tolist()[0]) -
                               (filled_arrPrev[index - 1].tolist()[0] - filled_arrPrevPrev[index - 1].tolist()[0]))
                dxdataq.append((filled_arrCurrent[index - 1].tolist()[1] - filled_arrPrev[index - 1].tolist()[1]) -
                               (filled_arrPrev[index - 1].tolist()[1] - filled_arrPrevPrev[index - 1].tolist()[1]))

                # Combine position and velocity for state
                xdata = xdatap + xdataq
                # Combine velocity and acceleration for derivatives
                dxdata = dxdatap + dxdataq

            dataset['x'].append(xdata)
            dataset['dx'].append(dxdata)

            # Update sliding window of previous rows
            prevrow.pop(0)
            prevrow.append(row)

        allData.append(dataset)
        # Split into train and test sets
        split_dict, test_dataset = split_data_with_shuffle(dataset)

        # Concatenate data from all masks
        x = torch.cat((x, torch.tensor(split_dict['x'], requires_grad=True, dtype=torch.float32)), dim=0)
        dxdt = torch.cat((dxdt, torch.tensor(split_dict['dx'], requires_grad=True, dtype=torch.float32)), dim=0)
        test_x = torch.cat((test_x, torch.tensor(test_dataset['x'], requires_grad=True, dtype=torch.float32)), dim=0)
        test_dxdt = torch.cat((test_dxdt, torch.tensor(test_dataset['dx'], requires_grad=True, dtype=torch.float32)),
                              dim=0)

    # Remove outliers from specific columns
    # Note: The column indices here seem to separate position and velocity features
    x = removeOutliyers(x, [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35])
    dxdt = removeOutliyers(dxdt, list(range(36)))  # Clip all columns for derivatives
    test_x = removeOutliyers(test_x, [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35])
    test_dxdt = removeOutliyers(test_dxdt, list(range(36)))

    # Visualize data distributions (raw data)
    x_np = x.detach().numpy()
    x_np2 = dxdt.detach().numpy()

    # Plot histograms for each feature (raw inputs)
    x_np_flat = x_np.reshape(-1, 36)  # Reshapes [samples, 36]
    plt.figure(figsize=(15, 10))
    for i in range(36):
        plt.subplot(6, 6, i + 1)
        plt.hist(x_np_flat[:, i], bins=30, color='skyblue', edgecolor='black')
        plt.title(f'Feature {i}')
        plt.xlabel('Value')
        plt.ylabel('Count')
    plt.tight_layout()
    plt.close()

    # Plot histograms for each feature (raw derivatives)
    x_np_flat2 = x_np2.reshape(-1, 36)
    plt.figure(figsize=(15, 10))
    for i in range(36):
        plt.subplot(6, 6, i + 1)
        plt.hist(x_np_flat2[:, i], bins=30, color='skyblue', edgecolor='black')
        plt.title(f'Feature {i}')
        plt.xlabel('Value')
        plt.ylabel('Count')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(2)  # Display for 2 seconds
    plt.close()

    # Alternative normalization approach (z-score normalization)
    mean = x.mean(dim=0)
    std = x.std(dim=0) + 1e-6  # Avoid division by zero
    inputs = (x - mean) / std
    targets = (dxdt - mean) / std  # Scale derivatives accordingly

    # Scale to [-1, 1] range (main normalization used)
    inputsScaled, mins1, maxs1 = scale_columns_neg1_pos1(x)
    targetsScaled, mins2, maxs2 = scale_columns_neg1_pos1(dxdt)

    # Visualize z-score normalized data
    inputs = inputs.detach().numpy()
    x_np_flat = inputs.reshape(-1, 36)
    plt.figure(figsize=(15, 10))
    for i in range(36):
        plt.subplot(6, 6, i + 1)
        plt.hist(x_np_flat[:, i], bins=30, color='skyblue', edgecolor='black')
        plt.title(f'Feature {i}')
        plt.xlabel('Value')
        plt.ylabel('Count')
    plt.tight_layout()
    plt.close()

    targets = targets.detach().numpy()
    x_np_flat2 = targets.reshape(-1, 36)
    plt.figure(figsize=(15, 10))
    for i in range(36):
        plt.subplot(6, 6, i + 1)
        plt.hist(x_np_flat2[:, i], bins=30, color='skyblue', edgecolor='black')
        plt.title(f'Feature {i}')
        plt.xlabel('Value')
        plt.ylabel('Count')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    # Scale test data using same scaling as training data
    test_x, mi, ma = scale_columns_neg1_pos1(test_x)
    test_dxdt, mii, maa = scale_columns_neg1_pos1(test_dxdt)

    # Visualize scaled training data
    inputsScalednp = inputsScaled.detach().numpy()
    x_np_flat = inputsScalednp.reshape(-1, 36)
    plt.figure(figsize=(15, 10))
    for i in range(36):
        plt.subplot(6, 6, i + 1)
        plt.hist(x_np_flat[:, i], bins=30, color='skyblue', edgecolor='black')
        plt.title(f'Feature {i}')
        plt.xlabel('Value')
        plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
    plt.close()

    targetsScalednp = targetsScaled.detach().numpy()
    x_np_flat2 = targetsScalednp.reshape(-1, 36)
    plt.figure(figsize=(15, 10))
    for i in range(36):
        plt.subplot(6, 6, i + 1)
        plt.hist(x_np_flat2[:, i], bins=30, color='skyblue', edgecolor='black')
        plt.title(f'Feature {i}')
        plt.xlabel('Value')
        plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
    plt.close()

    # Reshape data for neural network input
    inputsScaled = inputsScaled.reshape(-1, 36)
    targetsScaled = targetsScaled.reshape(-1, 36)
    test_x = test_x.reshape(-1, 36)
    test_dxdt = test_dxdt.reshape(-1, 36)

    return inputsScaled, targetsScaled, test_x, test_dxdt, mins1, maxs1, mins2, maxs2


def train(args):
    """
    Main training function for the Hamiltonian Neural Network (HNN) or baseline model.

    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments

    Returns:
    --------
    model : HNN or baseline model
        Trained model
    stats : dict
        Training statistics including loss values
    """
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.verbose = True

    # Initialize model and optimizer
    if args.verbose:
        print("Training baseline model:" if args.baseline else "Training HNN model:")

    # Determine output dimension based on model type
    output_dim = args.input_dim if args.baseline else 2
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model,
                field_type=args.field_type, baseline=args.baseline)

    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)
    # Note: Uncommented scheduler - could be used for learning rate decay
    # scheduler = StepLR(optim, step_size=200, gamma=0.1)

    # Load and prepare data
    x, dxdt, test_x, test_dxdt, min1, max1, min2, max2 = getMyDataAvg()

    # Training loop
    stats = {'train_loss': [], 'test_loss': []}
    for step in range(args.total_steps + 1):
        # Train step
        ixs = torch.randperm(x.shape[0])[:args.batch_size]
        dxdt_hat = model.time_derivative(x[ixs])
        loss = L2_loss(dxdt[ixs], dxdt_hat)
        loss.backward(retain_graph=True)

        # Calculate gradient norm for monitoring
        grad = torch.cat([p.grad.flatten() for p in model.parameters()]).clone()
        optim.step()
        optim.zero_grad()
        # scheduler.step()  # Uncomment if using learning rate scheduler

        # Evaluate on test data
        test_ixs = torch.randperm(test_x.shape[0])[:args.batch_size]
        test_dxdt_hat = model.time_derivative(test_x[test_ixs])
        test_loss = L2_loss(test_dxdt[test_ixs], test_dxdt_hat)

        # Logging
        stats['train_loss'].append(loss.item())
        stats['test_loss'].append(test_loss.item())

        if args.verbose and step % args.print_every == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}, grad norm {:.4e}, grad std {:.4e}"
                  .format(step, loss.item(), test_loss.item(), grad @ grad, grad.std()))

    # Plot training and test loss over time
    plt.figure(figsize=(10, 6))
    plt.plot(stats['train_loss'], label='Training loss')
    plt.plot(stats['test_loss'], label='Test loss')
    plt.xlabel('Training steps')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss over Time')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Use log scale for better visualization of wide-ranging losses
    plt.show()

    # Final evaluation
    train_dxdt_hat = model.time_derivative(x)
    train_dist = (dxdt - train_dxdt_hat) ** 2
    test_dxdt_hat = model.time_derivative(test_x)
    test_dist = (test_dxdt - test_dxdt_hat) ** 2

    print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
          .format(train_dist.mean().item(), train_dist.std().item() / np.sqrt(train_dist.shape[0]),
                  test_dist.mean().item(), test_dist.std().item() / np.sqrt(test_dist.shape[0])))

    return model, stats


if __name__ == "__main__":
    """
    Main execution block.
    """
    # Parse command line arguments
    args = get_args()
    args.baseline = False  # Force HNN training (not baseline)

    # Train model
    model, stats = train(args)

    # Save trained model
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    label = 'baseline' if args.baseline else 'hnn'
    model_path = '{}/{}-orbits9-{}.tar'.format(args.save_dir, args.name, label)
    torch.save(model.state_dict(), model_path)

    # Save training statistics
    stats_path = '{}/{}-orbits9-{}.pkl'.format(args.save_dir, args.name, label)
    to_pickle(stats, stats_path)