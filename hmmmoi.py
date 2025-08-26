import torch
import numpy as np
from nn_models import MLP
from hnn import HNN
from utils import L2_loss
from argparse import Namespace
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
import re
import ast
from data import get_dataset
args = Namespace(
    seed=0,
    input_dim=4,  # Updated automatically in function, but set for consistency
    hidden_dim=200,  # MLP hidden layer size
    nonlinearity='tanh',  # Activation function
    baseline=False,  # Use HNN, not baseline model
    field_type='solenoidal',  # Or 'gradient' depending on your HNN setup
    learn_rate=1e-3,
    total_steps=1000,
    print_every=100,
    verbose=True,
    use_rk4=True  # Use RK4 integration for better accuracy
)
def train(args, dataset, test_dataset, time_step):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Update input dimension for your dataset (4 for x, y, v_x, v_y per particle)
    args.input_dim = 4  # For one particle; use 4*N for N particles

    # Init model and optimizer
    if args.verbose:
        print("Training baseline model:" if args.baseline else "Training HNN model:")

    output_dim = args.input_dim if args.baseline else 2  # HNN outputs Hamiltonian gradients
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model,
                field_type=args.field_type, baseline=args.baseline)
    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)

    # Arrange data (replace get_dataset with your dataset)
    data = get_dataset(seed=args.seed)
    x = torch.tensor(data['x'], requires_grad=True, dtype=torch.float32)
    test_x = torch.tensor(data['test_x'], requires_grad=True, dtype=torch.float32)
    dxdt = torch.Tensor(data['dx'])
    test_dxdt = torch.Tensor(data['test_dx'])
    # Assume dataset is a dict or arrays with 'x' (state) and 'dx' (derivatives)
    #x = torch.tensor(dataset['x'], requires_grad=True, dtype=torch.float32)  # Shape: [N_samples, 4]
    #dxdt = torch.tensor(dataset['dx'], requires_grad=False, dtype=torch.float32)  # Shape: [N_samples, 4]
    #test_x = torch.tensor(test_dataset['x'], requires_grad=True, dtype=torch.float32)  # Test state
    #test_dxdt = torch.tensor(test_dataset['dx'], requires_grad=False, dtype=torch.float32)  # Test derivatives

    # Normalize data (optional, but recommended for stability)
    x_mean, x_std = x.mean(dim=0, keepdim=True), x.std(dim=0, keepdim=True) + 1e-6
    dxdt_mean, dxdt_std = dxdt.mean(dim=0, keepdim=True), dxdt.std(dim=0, keepdim=True) + 1e-6
    x = (x - x_mean) / x_std
    dxdt = (dxdt - dxdt_mean) / dxdt_std
    test_x = (test_x - x_mean) / x_std  # Use same normalization for test
    test_dxdt = (test_dxdt - dxdt_mean) / dxdt_std

    # Scale derivatives by time step (if dxdt represents change over \Delta t)
    dxdt = dxdt / time_step  # Convert to time derivatives
    test_dxdt = test_dxdt / time_step

    # Vanilla train loop
    stats = {'train_loss': [], 'test_loss': []}
    for step in range(args.total_steps + 1):
        # Train step
        print("tuka", x)
        dxdt_hat = model.rk4_time_derivative(x, dt=0.01) if args.use_rk4 else model.time_derivative(x, dt=0.01)
        loss = L2_loss(dxdt, dxdt_hat)
        loss.backward()
        optim.step()
        optim.zero_grad()

        # Run test data
        test_dxdt_hat = model.rk4_time_derivative(test_x, dt=0.01) if args.use_rk4 else model.time_derivative(test_x, dt=0.01)
        test_loss = L2_loss(test_dxdt, test_dxdt_hat)

        # Logging
        stats['train_loss'].append(loss.item())
        stats['test_loss'].append(test_loss.item())
        if args.verbose and step % args.print_every == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))

    # Final evaluation (undo normalization for reporting)
    train_dxdt_hat = model.time_derivative(x) * dxdt_std * time_step  # Rescale back
    train_dist = (dxdt * dxdt_std * time_step - train_dxdt_hat) ** 2
    test_dxdt_hat = model.time_derivative(test_x) * dxdt_std * time_step
    test_dist = (test_dxdt * dxdt_std * time_step - test_dxdt_hat) ** 2
    print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
          .format(train_dist.mean().item(), train_dist.std().item() / np.sqrt(train_dist.shape[0]),
                  test_dist.mean().item(), test_dist.std().item() / np.sqrt(test_dist.shape[0])))

    return model, stats

def parse_custom_array(s):
    # Remove np.float32() wrappers while keeping the numeric values
    cleaned = re.sub(r'np\.float32\(([^)]+)\)', r'\1', s)
    # Convert to actual list
    data = ast.literal_eval(cleaned)
    # Convert numbers to float32 if needed
    return [[x[0]] + [np.float32(y) for y in x[1:]] for x in data]


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
df = pd.read_csv(f'C:/Users/Alexs/PyCharmMiscProject/pythonProject7/object_tracking/object_tracking/samMooooolq/segmentAnything/videos16/mask1/trainData.csv')

# Display the DataFrame
print(df.head())
print(df['cap'])

rowindex = 0
prevrow = []
dataset = {}
dataset['x'] = []
dataset['dx'] = []
for row in df['cap']:
    row = parse_custom_array(row)
    if rowindex == 0:
        print(f"Raw row content: {row!r}")
        prevrow = row
        rowindex += 1
        print("prev", prevrow, type(prevrow))
        continue
    i = 0
    j = 0
    while i < len(prevrow) and j < len(row):
        if prevrow[i][0] == row[j][0]:
            dataset['x'].append(prevrow[i][1:])
            dataset['dx'].append(row[j][1:])
            i = i + 1
            j = j + 1
            continue
        if prevrow[i][0] < row[j][0] and i < len(prevrow):
            i = i + 1
            continue
        if prevrow[i][0] > row[j][0] and j < len(row):
            j = j + 1
            continue

#print(dataset)
def encode_pairs(dictionary):
    # Extract all numeric values to find maximum
    all_values = []
    for key in dictionary:
        for item in dictionary[key]:
            if isinstance(item, (list, np.ndarray)):
                all_values.extend(item)
            else:
                all_values.append(item)

    m = max(all_values)

    # Function to encode a list of values
    def encode_list(lst):
        encoded = []
        for i in range(0, len(lst), 2):  # Step by 2 to process pairs
            # Extract x and y values, handling both numpy and regular float cases
            x = lst[i][0] if isinstance(lst[i], (list, np.ndarray)) else lst[i]
            y = lst[i + 1][0] if isinstance(lst[i + 1], (list, np.ndarray)) else lst[i + 1]
            encoded.append(float(x) * m + float(y))
        return encoded

    return {
        'x': encode_list(dictionary['x']),
        'dx': encode_list(dictionary['dx'])
    }
#dataEncoded = encode_pairs(dataset)
split_dict, test_dataset= split_data(dataset)
print(len(split_dict['x']), len(test_dataset['x']), len(split_dict['dx']), len(test_dataset['dx']), split_dict['x'][0])
m = max(max(sublist) for sublist in sum(dataset.values(), []))
split_dict['x'] = [[row[0] * m + row[1], row[2] * 100 + row[3]] for row in split_dict['x']]
test_dataset['x'] = [[row[0] * m + row[1], row[2] * 100 + row[3]] for row in test_dataset['x']]
split_dict['dx'] = [[row[0] * m + row[1], row[2] * 100 + row[3]] for row in split_dict['dx']]
test_dataset['dx'] = [[row[0] * m + row[1], row[2] * 100 + row[3]] for row in test_dataset['dx']]
print(len(split_dict['x']), len(test_dataset['x']), len(split_dict['dx']), len(test_dataset['dx']), split_dict['x'][0])


model, stats = train(args, split_dict, test_dataset, 0.001)

'''
plt.plot(stats['train_loss'], label='Train Loss')
plt.plot(stats['test_loss'], label='Test Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()
plt.show()'''