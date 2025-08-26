# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch
import numpy as np

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
  elif name == 'elu':
    nl = torch.nn.functional.elu
  elif name == 'swish':
    nl = lambda x: x * torch.sigmoid(x)
  else:
    raise ValueError("nonlinearity not recognized")
  return nl
class MLP(torch.nn.Module):
  '''Just a salt-of-the-earth MLP'''
  def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh'):
    super(MLP, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
    self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear25 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=None)

    self.dropout1 = torch.nn.Dropout(0.3)
    self.dropout2 = torch.nn.Dropout(0.3)
    self.dropout3 = torch.nn.Dropout(0.3)

    for l in [self.linear1, self.linear2, self.linear3]:
      torch.nn.init.orthogonal_(l.weight) # use a principled initialization

    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def forward(self, x, separate_fields=False):
    h = self.nonlinearity( self.linear1(x) )
    h = self.dropout1(h)
    h = self.nonlinearity( self.linear2(h) )
    h = self.dropout2(h)
    h = self.nonlinearity( self.linear25(h) )
    h = self.dropout3(h)
    return self.linear3(h)

class MLPAutoencoder(torch.nn.Module):
  '''A salt-of-the-earth MLP Autoencoder + some edgy res connections'''
  def __init__(self, input_dim, hidden_dim, latent_dim, nonlinearity='tanh'):
    super(MLPAutoencoder, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
    self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear4 = torch.nn.Linear(hidden_dim, latent_dim)

    self.linear5 = torch.nn.Linear(latent_dim, hidden_dim)
    self.linear6 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear7 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear8 = torch.nn.Linear(hidden_dim, input_dim)

    for l in [self.linear1, self.linear2, self.linear3, self.linear4, \
              self.linear5, self.linear6, self.linear7, self.linear8]:
      torch.nn.init.orthogonal_(l.weight)  # use a principled initialization

    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def encode(self, x):
    h = self.nonlinearity( self.linear1(x) )
    h = h + self.nonlinearity( self.linear2(h) )
    h = h + self.nonlinearity( self.linear3(h) )
    return self.linear4(h)

  def decode(self, z):
    h = self.nonlinearity( self.linear5(z) )
    h = h + self.nonlinearity( self.linear6(h) )
    h = h + self.nonlinearity( self.linear7(h) )
    return self.linear8(h)

  def forward(self, x):
    z = self.encode(x)
    x_hat = self.decode(z)
    return x_hat