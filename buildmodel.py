# #创建一个神经网络
# 神经网络由层/模块组成，这些层/模块对数据进行操作。
# torch.nn 空间提供了构建自己的神经网络所需的所有构建模块。
# PyTorch 中的每个模块都是 nn.Module 的子类。神经网络本身就是一个模块，它由其他模块（层）组成。
# 这种嵌套结构使得构建和管理复杂的架构变得容易。
# 在接下来的部分中，我们将构建一个神经网络来对 FashionMNIST 数据集中的图像进行分类。
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# #我们希望能够在可用的硬件加速器上，如 GPU 或 MPS 上训练我们的模型。
# 让我们检查一下是否可用 torch.cuda 或 torch.backends.mps，否则我们使用 CPU。
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# #定义神经网络类
# 我们通过子类化 nn.Module 来定义我们的神经网络，并在 init 中初始化神经网络的层。
# 每个 nn.Module 的子类在 forward 方法中实现对输入数据的操作。
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

