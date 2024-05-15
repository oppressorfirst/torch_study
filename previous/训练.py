import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from model import *

import time

train_dataset = torchvision.datasets.CIFAR10("../dataset", train=True, transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor())

train_dataset_size = len(train_dataset)
test_dataset_size = len(test_dataset)

print(f"训练数据集的长度为{train_dataset_size}")
print(f"测试数据集的长度为{test_dataset_size}")

# 利用dataloader加载数据集
train_dataloader = DataLoader(train_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)

# 搭建模型
jl = JL()

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()

# 定义优化器
learning_rate = 1e-2
optim = torch.optim.SGD(jl.parameters(), lr=learning_rate)
start_time = time.time()
#  设置训练网络的参数
total_train_step = 0  # 训练次数
total_test_step = 0  # 测试次数
epoch = 1  # 训练次数

for i in range(epoch):
    print(f"第{i + 1}轮训练开始了")

    # 训练开始
    for data in train_dataloader:
        imgs, targets = data
        outputs = jl(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_train_step = total_train_step + 1
        end_time = time.time()
        if total_train_step % 100 == 0:
            print(end_time-start_time)
            print(f"训练次数{total_train_step},Loss:{loss.item()}")

    with torch.no_grad():
        total_test_loss = 0
        for data in test_dataloader:
            imgs, targets = data
            outputs = jl(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss

    print(f"整体测试集上的Loss:{total_test_loss}")

    torch.save(jl, f"jl_{i}")
    print("模型已经保存")
