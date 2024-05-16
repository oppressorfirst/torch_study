import os
import torch
import torchvision.transforms
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


data = torch.randn(4,3)
print(f"data is \n{data}")
print()
x_data = torch.tensor(data)

flatten_layer = nn.Flatten()
result = flatten_layer(x_data)
print(x_data)
print(result)
