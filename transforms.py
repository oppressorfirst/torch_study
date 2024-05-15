# #数据并不总是以适合训练机器学习算法的最终处理形式出现。我们使用转换来对数据进行一些操作，使其适合训练。
# 所有 TorchVision 数据集都有两个参数 - transform 用于修改训练用的数据（图片）
# target_transform 用于修改标签 - 它们接受包含转换逻辑的可调用对象。torchvision.transforms 模块提供了几种常用的转换。
# FashionMNIST 的特征以 PIL 图像格式表示，标签为整数。为了训练，我们需要将特征转换为标准化的张量，将标签转换为 one-hot 编码的张量。为了进行这些转换，我们使用 ToTensor 和 Lambda。
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    # ToTensor()将 PIL 图像或 NumPy 数组转换为 FloatTensor，并将图像的像素强度值缩放到 [0., 1.] 范围内。
    transform=ToTensor(),
    # Lambda 转换应用任何用户定义的 lambda 函数。在这里，我们定义一个函数将整数转换为一个 one-hot 编码的张量。
    # 它首先创建一个大小为 10 的零张量（我们数据集中标签的数量），然后调用 scatter_ 函数，该函数在标签 y 给定的索引上分配值为 1。
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

