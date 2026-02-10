# Import necessary libraries
import torch
import os
import sys
import argparse

# Add parent directory to path to import custom modules
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
sys.path.append(PARENT_DIR)

# Import custom modules - HNN (Hamiltonian Neural Network) and MLP (Multi-Layer Perceptron)
from hnn import HNN, MLP


# Note: The following imports are commented out as they might not be available in current scope
# from TrainHNNModel import xy_to_hamiltonian_qp, hamiltonian_qp_to_xy


def get_args():
    """
    Parse command line arguments for the 4D model configuration (2D position + 2D velocity).

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='HNN Model Configuration')
    parser.add_argument('--input_dim', default=4, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=400, type=int, help='hidden dimension of mlp')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=200, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=20, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='spring', type=str, help='model name')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--use_rk4', dest='use_rk4', action='store_true', help='integrate derivative with RK4')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--field_type', default='solenoidal', type=str, help='type of vector field to learn')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.set_defaults(feature=True)
    return parser.parse_args()


def get_args9():
    """
    Parse command line arguments for the 36D model configuration (9 points × 4 dimensions each).

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='HNN Model Configuration - 36D')
    parser.add_argument('--input_dim', default=36, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=1200, type=int, help='hidden dimension of mlp')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=600, type=int, help='batch_size')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=400, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='cleanPerf', type=str, help='model name')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--field_type', default='solenoidal', type=str, help='type of vector field to learn')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.set_defaults(feature=True)
    return parser.parse_args()


# Global model variable (currently unused in the provided code)
globalModel = None


def HNNPredict(x, y, vx, vy, local=False):
    """
    Predict using the 2D HNN model with position and velocity inputs.
    This function is designed for a 2D spring system.

    Args:
        x: x position
        y: y position
        vx: x velocity
        vy: y velocity
        local: whether to load model from local directory (True) or from codeFromPaperHnn/ (False)

    Returns:
        List of predicted derivatives in xy coordinates [dx/dt, dy/dt, dvx/dt, dvy/dt]
    """
    # Get default arguments for 4D model
    args = get_args()
    args.baseline = False
    args.verbose = True

    # Define model architecture
    # For HNN, output_dim is 2 (Hamiltonian gradient), for baseline it's 4 (direct prediction)
    output_dim = args.input_dim if args.baseline else 2
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model,
                field_type=args.field_type, baseline=args.baseline)

    # Load pre-trained model weights
    model_path = 'spring-hnn.tar' if local else 'codeFromPaperHnn/spring-hnn.tar'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)

    # Note: The following conversion functions are commented out in imports
    # They would convert between Cartesian coordinates and Hamiltonian (q,p) coordinates
    point, revert = xy_to_hamiltonian_qp(x, y, vx, vy)

    # Prepare input tensor for the model
    torch_tensor = torch.tensor(point, requires_grad=True).reshape(1, 2)

    # Get time derivative from HNN model
    dxdt_hat = model.time_derivative(torch_tensor)

    # Convert back to xy coordinates
    val1, val2 = dxdt_hat[0]
    bdx = hamiltonian_qp_to_xy(val1.item(), val2.item(), revert[0], revert[1])

    return list(bdx)


def HNNCleanPredict(x, y, px, py, local=False):
    """
    Predict using the 4D HNN model with position and momentum inputs.
    This function performs multiple integration steps.

    Args:
        x: x position
        y: y position
        px: x momentum
        py: y momentum
        local: whether to load model from local directory

    Returns:
        Tuple of new x and y positions after three integration steps
    """
    # Get default arguments for 4D model
    args = get_args()
    args.baseline = False
    args.verbose = True

    # Define model architecture
    output_dim = args.input_dim if args.baseline else 2
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model,
                field_type=args.field_type, baseline=args.baseline)

    # Load pre-trained model weights
    model_path = 'cleanPerf-orbits2-hnn.tar'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)

    # Normalization parameters for input and output scaling
    # These values appear to be min-max ranges for denormalization
    min1 = torch.tensor([2.6857, 2.8891, -60.0000, -50.0000])
    max1 = torch.tensor([460.8437, 258.8189, 80.0000, 60.0000])
    min2 = torch.tensor([-60., -50., -150., -150.])
    max2 = torch.tensor([80., 60., 150., 150.])

    # Prepare initial state tensor
    torch_tensor_original = torch.tensor([x, y, px, py], requires_grad=True).reshape(1, 4)

    # Reshape normalization tensors for broadcasting
    min1 = min1.reshape(1, 4)
    max1 = max1.reshape(1, 4)
    min2 = min2.reshape(1, 4)
    max2 = max2.reshape(1, 4)

    # Perform three integration steps using the model
    for _ in range(3):
        # Normalize input to [-1, 1] range
        torch_tensor = 2 * (torch_tensor_original - min1) / (max1 - min1) - 1

        # Get derivatives from model
        dxdt_hat = model.time_derivative(torch_tensor)

        # Denormalize the output derivatives
        dxdt_hat = min2 + (dxdt_hat + 1) * (max2 - min2) / 2

        # Update positions and momenta using semi-implicit Euler integration
        # Position update includes half of velocity contribution
        nx = x + dxdt_hat[0][0] + dxdt_hat[0][2] / 2
        ny = y + dxdt_hat[0][1] + dxdt_hat[0][3] / 2

        # Momentum update (full step)
        npx = px + dxdt_hat[0][2]
        npy = py + dxdt_hat[0][3]

        # Prepare for next iteration
        torch_tensor_original = torch.tensor([nx, ny, npx, npy], requires_grad=True)
        x, y, px, py = nx, ny, npx, npy

    return nx, ny


def NinePointPredict(torch_tensor_original_arr, local=True):
    """
    Predict using the 36D HNN model for 9 points (each with position and velocity).
    This models a system of 9 interacting particles in 2D.

    Args:
        torch_tensor_original_arr: Input tensor array with 36 elements
                                   Format: [x1, y1, x2, y2, ..., x9, y9,
                                           vx1, vy1, vx2, vy2, ..., vx9, vy9]
        local: whether to load model from local directory

    Returns:
        List of predicted values after three integration steps
    """
    # Get arguments for 36D model
    args = get_args9()
    output_dim = args.input_dim if args.baseline else 2

    # Define model architecture
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model,
                field_type=args.field_type, baseline=args.baseline)

    # Normalization parameters for 36D system
    # These are min-max ranges for the 9-point system
    min1 = torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                          0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                          0.0000, 0.0000, 0.0000, 0.0000, -18.3632, -11.2353, -15.6455,
                          -7.9202, -16.7414, -8.2901, -18.3238, -8.8134, -13.6065, -9.1661,
                          -16.6068, -9.3443, -19.4017, -9.1756, -15.3719, -10.9581, -17.1059,
                          -11.4423]])
    max1 = torch.tensor([[457.4383, 258.2751, 452.5234, 258.0087, 454.1888, 244.3642, 460.8437,
                          258.8189, 431.3252, 255.6370, 455.2306, 253.5110, 464.9813, 252.8056,
                          451.5783, 258.1815, 447.0760, 256.4935, 40.8561, 25.1112, 31.7861,
                          15.9433, 40.3561, 18.2105, 35.9572, 15.6122, 30.7770, 21.8802,
                          36.3843, 20.2653, 50.9019, 22.2659, 28.4258, 19.9505, 39.2759,
                          27.2650]])
    min2 = torch.tensor([[-18.3632, -11.2353, -15.6455, -7.9202, -16.7414, -8.2901, -18.3238,
                          -8.8134, -13.6065, -9.1661, -16.6068, -9.3443, -19.4017, -9.1756,
                          -15.3719, -10.9581, -17.1059, -11.4423, -44.6970, -27.1456, -36.3283,
                          -18.5835, -42.9202, -19.4917, -45.0034, -21.0475, -33.9981, -21.8763,
                          -36.6700, -20.8943, -50.8903, -22.0966, -36.7851, -26.3344, -42.4399,
                          -30.5868]])
    max2 = torch.tensor([[40.8561, 25.1112, 31.7861, 15.9433, 40.3561, 18.2105, 35.9572, 15.6122,
                          30.7770, 21.8802, 36.3843, 20.2653, 50.9019, 22.2659, 28.4258, 19.9505,
                          39.2759, 27.2650, 62.2126, 36.8080, 49.6731, 23.9610, 57.2762, 25.6507,
                          58.0214, 25.7220, 45.1554, 31.7578, 49.1985, 28.0326, 72.1932, 30.7016,
                          46.0045, 32.1513, 59.8764, 41.6628]])

    # Load pre-trained model weights
    model_path = 'cleanPerf-orbits9-hnn.tar' if local else 'cleanPerf-orbits9-hnn.tar'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)

    # Prepare input tensor
    torch_tensor_original = torch.tensor(torch_tensor_original_arr, requires_grad=True).reshape(1, 36)

    # Reshape normalization tensors
    min1 = min1.reshape(1, 36)
    max1 = max1.reshape(1, 36)
    min2 = min2.reshape(1, 36)
    max2 = max2.reshape(1, 36)

    # Perform three integration steps
    for step in range(3):
        # Normalize input to [-1, 1] range
        torch_tensor = 2 * (torch_tensor_original - min1) / (max1 - min1) - 1

        # Get derivatives from model
        dxdt_hat = model.time_derivative(torch_tensor)

        # Denormalize the output derivatives
        dxdt_hat = min2 + (dxdt_hat + 1) * (max2 - min2) / 2

        if step < 2:  # For first two steps, update the state
            # Split into position (first 18) and velocity (last 18) components
            first_half1 = torch_tensor_original[:, :18]  # positions (x,y for 9 points)
            second_half1 = torch_tensor_original[:, 18:]  # velocities (vx,vy for 9 points)
            first_half2 = dxdt_hat[:, :18]  # position derivatives (dx/dt, dy/dt)
            second_half2 = dxdt_hat[:, 18:]  # velocity derivatives (dvx/dt, dvy/dt)

            # Update state using semi-implicit Euler integration
            # Positions: x_new = x_old + dx/dt + 0.5 * dv/dt
            # Velocities: v_new = v_old + dv/dt
            new_tensor = torch.cat([
                first_half1 + first_half2 + (0.5 * second_half2),  # update positions
                second_half1 + second_half2  # update velocities
            ], dim=1)

            torch_tensor_original = new_tensor

    # Return final state as a list
    return torch_tensor_original.tolist()