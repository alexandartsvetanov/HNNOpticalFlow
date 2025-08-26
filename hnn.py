# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch
import numpy as np

from codeFromPaperHnn.nn_models import MLP
from codeFromPaperHnn.utils import rk4


class HNN(torch.nn.Module):
    '''Learn arbitrary vector fields that are sums of conservative and solenoidal fields'''

    def __init__(self, input_dim, differentiable_model, field_type='solenoidal',
                 baseline=False, assume_canonical_coords=True):
        super(HNN, self).__init__()
        self.baseline = baseline
        self.differentiable_model = differentiable_model
        self.assume_canonical_coords = assume_canonical_coords
        self.M = self.permutation_tensor(input_dim)  # Levi-Civita permutation tensor
        self.field_type = field_type

    def forward(self, x):
        # traditional forward pass
        if self.baseline:
            return self.differentiable_model(x)

        y = self.differentiable_model(x)
        #print("Dims", y.dim(), y.shape, y[0], x[0])
        assert y.dim() == 2 and y.shape[1] == 2, "Output tensor should have shape [batch_size, 2]"
        return y.split(1, 1)

    def rk4_time_derivative(self, x, dt):
        return rk4(fun=self.time_derivative, y0=x, t=0, dt=dt)

    def time_derivative(self, x, t=None, separate_fields=False):
        '''NEURAL ODE-STLE VECTOR FIELD'''
        if self.baseline:
            return self.differentiable_model(x)

        '''NEURAL HAMILTONIAN-STLE VECTOR FIELD'''
        F1, F2 = self.forward(x)  # traditional forward pass

        conservative_field = torch.zeros_like(x)  # start out with both components set to 0
        solenoidal_field = torch.zeros_like(x)

        if self.field_type != 'solenoidal':
            dF1 = torch.autograd.grad(F1.sum(), x, create_graph=True)[0]  # gradients for conservative field
            conservative_field = dF1 @ torch.eye(*self.M.shape)

        if self.field_type != 'conservative':
            dF2 = torch.autograd.grad(F2.sum(), x, create_graph=True)[0]  # gradients for solenoidal field
            solenoidal_field = dF2 @ self.M.t()

        if separate_fields:
            return [conservative_field, solenoidal_field]

        return conservative_field + solenoidal_field

    def permutation_tensor(self, n):
        M = None
        if self.assume_canonical_coords:
            M = torch.eye(n)
            M = torch.cat([M[n // 2:], -M[:n // 2]])
        else:
            '''Constructs the Levi-Civita permutation tensor'''
            M = torch.ones(n, n)  # matrix of ones
            M *= 1 - torch.eye(n)  # clear diagonals
            M[::2] *= -1  # pattern of signs
            M[:, ::2] *= -1

            for i in range(n):  # make asymmetric
                for j in range(i + 1, n):
                    M[i, j] *= -1
        return M


class PixelHNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, autoencoder,
                 field_type='solenoidal', nonlinearity='tanh', baseline=False):
        super(PixelHNN, self).__init__()
        self.autoencoder = autoencoder
        self.baseline = baseline

        output_dim = input_dim if baseline else 2
        nn_model = MLP(input_dim, hidden_dim, output_dim, nonlinearity)
        self.hnn = HNN(input_dim, differentiable_model=nn_model, field_type=field_type, baseline=baseline)

    def encode(self, x):
        return self.autoencoder.encode(x)

    def decode(self, z):
        return self.autoencoder.decode(z)

    def time_derivative(self, z, separate_fields=False):
        return self.hnn.time_derivative(z, separate_fields)

    def forward(self, x):
        z = self.encode(x)
        z_next = z + self.time_derivative(z)
        return self.decode(z_next)


import torch
from torchdiffeq import odeint as rk4  # Assuming rk4 is imported from torchdiffeq or similar

import torch

def choose_nonlinearity(name):
    nl = None
    if name == 'tanh':
        nl = torch.tanh
    elif name == 'relu':
        nl = torch.relu
    elif name == 'sigmoid':
        nl = torch.sigmoid
    elif name == 'softplus':
        nl = torch.nn.functional.softplus
    elif name == 'selu':
        nl = torch.nn.functional.selu
    elif name == 'swish':
        nl = lambda x: x * torch.sigmoid(x)
    else:
        raise ValueError("nonlinearity not recognized")
    return nl

class MLPMoi(torch.nn.Module):
    '''Just a salt-of-the-earth MLP'''
    def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh'):
        super(MLPMoi, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=None)

        for l in [self.linear1, self.linear2, self.linear3]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization

        self.nonlinearity = choose_nonlinearity(nonlinearity)

    def forward(self, x, separate_fields=False):
        h = self.nonlinearity(self.linear1(x))
        h = self.nonlinearity(self.linear2(h))
        return self.linear3(h)

class HNNMoi(torch.nn.Module):
    '''Learn arbitrary vector fields that are sums of conservative and solenoidal fields'''
    def __init__(self, input_dim, differentiable_model, field_type='solenoidal',
                 baseline=False, assume_canonical_coords=True):
        super(HNNMoi, self).__init__()
        self.baseline = baseline
        self.differentiable_model = differentiable_model
        self.assume_canonical_coords = assume_canonical_coords
        self.M = self.permutation_tensor(input_dim)  # Levi-Civita permutation tensor
        self.field_type = field_type

    def forward(self, x):
        # traditional forward pass
        if self.baseline:
            return self.differentiable_model(x)

        y = self.differentiable_model(x)
        assert y.dim() == 2 and y.shape[1] == 2, "Output tensor should have shape [batch_size, 2]"
        return y.split(1, 1)

    def rk4_time_derivative(self, x, dt):
        return rk4(fun=self.time_derivative, y0=x, t=0, dt=dt)

    def time_derivative(self, x, t=None, separate_fields=False):
        '''NEURAL ODE-STYLE VECTOR FIELD'''
        if self.baseline:
            return self.differentiable_model(x)

        '''NEURAL HAMILTONIAN-STYLE VECTOR FIELD'''
        F1, F2 = self.forward(x)  # traditional forward pass

        conservative_field = torch.zeros_like(x)  # start out with both components set to 0
        solenoidal_field = torch.zeros_like(x)

        if self.field_type != 'solenoidal':
            dF1 = torch.autograd.grad(F1.sum(), x, create_graph=True)[0]  # gradients for conservative field
            conservative_field = dF1 @ torch.eye(*self.M.shape)

        if self.field_type != 'conservative':
            dF2 = torch.autograd.grad(F2.sum(), x, create_graph=True)[0]  # gradients for solenoidal field
            solenoidal_field = dF2 @ self.M.t()

        if separate_fields:
            return [conservative_field, solenoidal_field]

        return conservative_field + solenoidal_field

    def permutation_tensor(self, n):
        M = None
        if self.assume_canonical_coords:
            # For two pairs of canonical coordinates (q1, p1, q2, p2), create block-diagonal symplectic matrix
            M = torch.zeros(n, n)
            block = torch.tensor([[0, 1], [-1, 0]])
            M[:2, :2] = block
            M[2:, 2:] = block
        else:
            # General Levi-Civita permutation tensor for 4D
            M = torch.ones(n, n)  # matrix of ones
            M *= 1 - torch.eye(n)  # clear diagonals
            M[::2] *= -1  # pattern of signs
            M[:, ::2] *= -1
            for i in range(n):  # make asymmetric
                for j in range(i + 1, n):
                    M[i, j] *= -1
        return M