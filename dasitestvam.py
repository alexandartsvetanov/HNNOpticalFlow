import torch
import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from codeFromPaperHnn.hnn import HNN, MLP
import torch, argparse

from codeFromPaperHnn.trainmoi import  xy_to_hamiltonian_qp, hamiltonian_qp_to_xy

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=4, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=400, type=int, help='hidden dimension of mlp')
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
def get_args9():
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
globalModel = None
def HNNPredict(x, y, vx, vy, local):
    print(local)
    args = get_args()
    #print(args, args.use_rk4)
    args.baseline = False
    args.verbose = True
    # 1. Define your model architecture (must match the saved model)
    output_dim = args.input_dim if args.baseline else 2
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model,
                field_type=args.field_type, baseline=args.baseline)
    globalModel = model

    # 2. Load the saved state dictionary
    print(local)
    if local == True:
        checkpoint = torch.load('spring-hnn.tar')  # or 'model.pth.tar'
    else:
        checkpoint = torch.load('codeFromPaperHnn/spring-hnn.tar')  # or 'model.pth.tar'

    #print(checkpoint)
    model.load_state_dict(checkpoint)
    # 3. Load weights into model
    model.load_state_dict(checkpoint)

    point, revert = xy_to_hamiltonian_qp(x, y, vx, vy)
    torch_tensor = torch.tensor(point, requires_grad=True)
    torch_tensor = torch_tensor.reshape(1, 2)
    dxdt_hat = model.time_derivative(torch_tensor)
    val1, val2 = dxdt_hat[0]
    bdx = hamiltonian_qp_to_xy(val1.item(), val2.item(), revert[0], revert[1])
    return list(bdx)


def HNNPredictAngle(x, y, vx, vy, local=False):

    args = get_args()
    #print(args, args.use_rk4)
    args.baseline = False
    args.verbose = True
    # 1. Define your model architecture (must match the saved model)
    output_dim = args.input_dim if args.baseline else 2
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model,
                field_type=args.field_type, baseline=args.baseline)

    # 2. Load the saved state dictionary
    if local == True:
        checkpoint = torch.load('spring-hnn.tar')  # or 'model.pth.tar'
    else:
        checkpoint = torch.load('codeFromPaperHnn/spring-hnn.tar')  # or 'model.pth.tar'

    #print(checkpoint)
    model.load_state_dict(checkpoint)
    # 3. Load weights into model
    model.load_state_dict(checkpoint)

    point, revert = xy_to_hamiltonian_qp(x, y, vx, vy)
    torch_tensor = torch.tensor(point, requires_grad=True)
    torch_tensor = torch_tensor.reshape(1, 2)
    dxdt_hat = model.time_derivative(torch_tensor)
    val1, val2 = dxdt_hat[0]
    bdx = hamiltonian_qp_to_xy(val1.item(), val2.item(), revert[0], revert[1])
    return list(bdx)

def HNNCleanPredict(x, y, px, py, local=False):
    args = get_args()
    #print(args, args.use_rk4)
    args.baseline = False
    args.verbose = True
    # 1. Define your model architecture (must match the saved model)

    output_dim = args.input_dim if args.baseline else 2
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model,
                field_type=args.field_type, baseline=args.baseline)

    # 2. Load the saved state dictionary
    if local == True:
        checkpoint = torch.load('codeFromPaperHnn/cleanPerf-orbits2-hnn.tar')  # or 'model.pth.tar'
    else:
        checkpoint = torch.load('codeFromPaperHnn/cleanPerf-orbits2-hnn.tar')  # or 'model.pth.tar'

    #print(checkpoint)
    model.load_state_dict(checkpoint)
    # 3. Load weights into model
    model.load_state_dict(checkpoint)

    torch_tensororiginal = torch.tensor([x, y, px, py], requires_grad=True)
    min1 = torch.tensor([  2.6857,   2.8891, -60.0000, -50.0000])
    max1 = torch.tensor([460.8437, 258.8189,  80.0000,  60.0000])
    min2 = torch.tensor([ -60.,  -50., -150., -150.])
    max2 = torch.tensor([ 80.,  60., 150., 150.])

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
    dxdt_hat = min2 + (dxdt_hat + 1) * (max2 - min2) / 2
    print("Output in coordinates", dxdt_hat[0])
    nx = x + dxdt_hat[0][0] + dxdt_hat[0][2] / 2
    ny = y + dxdt_hat[0][1] + dxdt_hat[0][3] / 2
    npx = px + dxdt_hat[0][2]
    npy = py + dxdt_hat[0][3]
    torch_tensororiginal = torch.tensor([nx, ny, npx, npy], requires_grad=True)
    print("Original in coordinates", torch_tensororiginal)
    torch_tensor = 2 * (torch_tensororiginal - min1) / (max1 - min1) - 1
    print("Normalized original", torch_tensor)
    dxdt_hat = model.time_derivative(torch_tensor)
    print("Normalized output", dxdt_hat)
    dxdt_hat = min2 + (dxdt_hat + 1) * (max2 - min2) / 2
    print("Output in coordinates", dxdt_hat[0])
    nx = x + dxdt_hat[0][0] + dxdt_hat[0][2] / 2
    ny = y + dxdt_hat[0][1] + dxdt_hat[0][3] / 2
    npx = px + dxdt_hat[0][2]
    npy = py + dxdt_hat[0][3]

    torch_tensororiginal = torch.tensor([nx, ny, npx, npy], requires_grad=True)
    print("Original in coordinates", torch_tensororiginal)
    torch_tensor = 2 * (torch_tensororiginal - min1) / (max1 - min1) - 1
    print("Normalized original", torch_tensor)
    dxdt_hat = model.time_derivative(torch_tensor)
    print("Normalized output", dxdt_hat)
    dxdt_hat = min2 + (dxdt_hat + 1) * (max2 - min2) / 2
    print("Output in coordinates", dxdt_hat[0])
    nx = x + dxdt_hat[0][0] + dxdt_hat[0][2] / 2
    ny = y + dxdt_hat[0][1] + dxdt_hat[0][3] / 2
    npx = px + dxdt_hat[0][2]
    npy = py + dxdt_hat[0][3]
    return nx, ny

def NinePointPredict(torch_tensororiginalArr, local = True):
    args = get_args9()
    output_dim = args.input_dim if args.baseline else 2
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model,
                field_type=args.field_type, baseline=args.baseline)
    min1 = torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
             0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
             0.0000, 0.0000, 0.0000, 0.0000, -18.3632, -11.2353, -15.6455,
             -7.9202, -16.7414, -8.2901, -18.3238, -8.8134, -13.6065, -9.1661,
             -16.6068, -9.3443, -19.4017, -9.1756, -15.3719, -10.9581, -17.1059,
             -11.4423]])
    max1 = torch.tensor(
        [[457.4383, 258.2751, 452.5234, 258.0087, 454.1888, 244.3642, 460.8437,
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
    max2 = torch.tensor(
        [[40.8561, 25.1112, 31.7861, 15.9433, 40.3561, 18.2105, 35.9572, 15.6122,
          30.7770, 21.8802, 36.3843, 20.2653, 50.9019, 22.2659, 28.4258, 19.9505,
          39.2759, 27.2650, 62.2126, 36.8080, 49.6731, 23.9610, 57.2762, 25.6507,
          58.0214, 25.7220, 45.1554, 31.7578, 49.1985, 28.0326, 72.1932, 30.7016,
          46.0045, 32.1513, 59.8764, 41.6628]])

    if local == True:
        checkpoint = torch.load('cleanPerf-orbits9-hnn.tar')  # or 'model.pth.tar'
    else:
        checkpoint = torch.load('codeFromPaperHnn/cleanPerf-orbits9-hnn.tar')  # or 'model.pth.tar'

    #print(checkpoint)
    model.load_state_dict(checkpoint)
    # 3. Load weights into model
    model.load_state_dict(checkpoint)

    torch_tensororiginal = torch.tensor(torch_tensororiginalArr, requires_grad=True)

    torch_tensororiginal = torch_tensororiginal.reshape(1, 36)
    min1 = min1.reshape(1, 36)
    max1 = max1.reshape(1, 36)
    min2 = min2.reshape(1, 36)
    max2 = max2.reshape(1, 36)

    print(min1, max1)
    print(min2, max2)
    print("Original in coordinates", torch_tensororiginal)
    torch_tensor = 2 * (torch_tensororiginal - min1) / (max1 - min1) - 1
    print("Normalized original", torch_tensor)
    dxdt_hat = model.time_derivative(torch_tensor)
    print("Normalized output", dxdt_hat)
    dxdt_hat = min2 + (dxdt_hat + 1) * (max2 - min2) / 2
    print("###############################")
    print("Output in coordinates", dxdt_hat)  #
    print("###############################")

    for i in range(2):
        first_half1 = torch_tensororiginal[:, :18]

        second_half1 = torch_tensororiginal[:, 18:]
        print("First half", first_half1, second_half1)
        first_half2 = dxdt_hat[:, :18]
        second_half2 = dxdt_hat[:, 18:]
        print("Second half", second_half2)

        # Create the new tensor
        new_tensor = torch.cat([
            first_half1 + first_half2 + (0.5 * second_half2),
            second_half1 + second_half2
        ], dim=1)
        torch_tensororiginal = new_tensor

        print("Original in coordinates", new_tensor)
        torch_tensor = 2 * (new_tensor - min1) / (max1 - min1) - 1
        print("Normalized original", torch_tensor)
        dxdt_hat = model.time_derivative(torch_tensor)
        print("Normalized output", dxdt_hat)
        dxdt_hat = min2 + (dxdt_hat + 1) * (max2 - min2) / 2
        print("###############################")
        print("Output in coordinates", dxdt_hat)  #
        print("###############################")

    first_half1 = torch_tensororiginal[:, :18]

    second_half1 = torch_tensororiginal[:, 18:]
    print("First half", first_half1, second_half1)
    first_half2 = dxdt_hat[:, :18]
    second_half2 = dxdt_hat[:, 18:]
    print("Second half", second_half2)

    # Create the new tensor
    new_tensor = torch.cat([
        first_half1 + first_half2 + (0.5 * second_half2),
        second_half1 + second_half2
    ], dim=1)
    tensor_list = new_tensor.tolist()

    return tensor_list
#a = HNNCleanPredict(282.6668, 80.6123, 0.4338, 2.1671)
'''a = NinePointPredict([82.6668, 180.6123, 82.6668, 180.6123, 82.6668, 180.6123, 82.6668, 180.6123, 82.6668, 180.6123,
         82.6668, 180.6123, 82.6668, 180.6123, 82.6668, 180.6123, 82.6668, 180.6123, 12.4338, -5.1671
            , 12.4338, -5.1671, 12.4338, -5.1671, 12.4338, -5.1671, 12.4338, -5.1671, 12.4338, -5.1671
            , 12.4338, -5.1671, 12.4338, -5.1671, 12.4338, -5.1671])
print(a)'''
#a = HNNCleanPredict(248.0, 200.0, 159.0, 38.0, True)
'''a, ra = xy_to_hamiltonian_qp(300, 300, -0.5, -0.5)  # , xy_to_hamiltonian_qp(100, 100, 0.7, 0.7)
print(a)
b = hamiltonian_qp_to_xy(a[0], a[1], ra[0], ra[1])
print(b)
torch_tensor = torch.tensor(a, requires_grad=True)
torch_tensor = torch_tensor.reshape(1, 2)
print("Posledno", torch_tensor)
dxdt_hat = globalModel.time_derivative(torch_tensor)
print(dxdt_hat)
val1, val2 = dxdt_hat[0]
bdx = hamiltonian_qp_to_xy(val1.item(), val2.item(), 345.2535300326414, 0.5792844463634922)
print(bdx)'''

#print("Chislata", HNNPredict(426, 144, -13, 7, True))
#print("Chislata", HNNPredict(326, 244, 8, -2, True))