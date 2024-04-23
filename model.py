import math
import copy
import time
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torch.optim as optim
from tqdm import tqdm
from functools import partial
from matplotlib import cm
from scipy.stats import qmc
import matplotlib.pyplot as plt
from pyDOE import lhs
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List, Callable, Union



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PINN_Net(nn.Module):
    def __init__(self, PDE: str, hidden_sizes: List[int], ub: List[Union[int, float]], lb: List[Union[int, float]], activation: Callable):
        super(PINN_Net, self).__init__()
        
        if PDE == 'Navier_Stokes':
            self.input_size = 2
            self.output_size = 6
        elif PDE == 'Heat_Conduction':
            self.input_size = 2
            self.output_size = 1
        elif PDE == 'Burger':
            self.input_size = 2
            self.output_size = 1
        else: 
            raise ValueError("Unsupported PDE type")

        self.hidden_sizes = hidden_sizes
        self.act = activation
        self.ub = torch.tensor(ub, dtype=torch.float32).to(DEVICE)
        self.lb = torch.tensor(lb, dtype=torch.float32).to(DEVICE)

        # Input layer
        self.fc1 = nn.Linear(self.input_size, self.hidden_sizes[0])
        nn.init.xavier_uniform_(self.fc1.weight)
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for k in range(len(hidden_sizes)-1):
            layer = nn.Linear(hidden_sizes[k], hidden_sizes[k+1])
            nn.init.xavier_uniform_(layer.weight)
            self.hidden_layers.append(layer)
        # Output layer
        self.fc2 = nn.Linear(self.hidden_sizes[-1], self.output_size)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input layer
        x = (x - self.lb) / (self.ub - self.lb)
        out = x
        out = self.act(self.fc1(out))
        # Hidden layers
        for layer in self.hidden_layers:
            out = self.act(layer(out))
        # Output layer
        out = self.fc2(out)
        return out
