import torch
from torch import nn
from torch.nn import ReLU

input = torch.tensor([[1, -0.5],
                      [2, 4]])

input = torch.reshape(input, (-1, 1, 2, 2))


class JL(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = ReLU()

    def forward(self, x):
        output = self.relu1(x)
        return output


jl = JL()
print(jl(input))
print(input.shape)
input2 = torch.randn(128, 20)
print(input2.shape)



