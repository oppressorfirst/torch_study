import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from model import *

import time

# 检查 MPS 是否可用并设置设备
if torch.backends.mps.is_available():
    device = torch.device("mps")  # 设置设备为 MPS
    print("Using MPS for acceleration")
else:
    device = torch.device("cpu")  # 回退到 CPU
    print("MPS not available, using CPU")

# 加载数据集
train_dataset = torchvision.datasets.CIFAR10("../dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

train_dataset_size = len(train_dataset)
test_dataset_size = len(test_dataset)

print(f"训练数据集的长度为 {train_dataset_size}")
print(f"测试数据集的长度为 {test_dataset_size}")

# 利用 DataLoader 加载数据集
train_dataloader = DataLoader(train_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)

# 搭建模型
jl = JL().to(device)  # 将模型移动到指定设备

# 创建损失函数
loss_fn = nn.CrossEntropyLoss().to(device)  # 将损失函数移动到指定设备

# 定义优化器
learning_rate = 1e-2
optim = torch.optim.SGD(jl.parameters(), lr=learning_rate)

# 设置训练网络的参数
total_train_step = 0  # 训练次数
total_test_step = 0  # 测试次数
epoch = 1  # 训练轮数

for i in range(epoch):
    print(f"第 {i + 1} 轮训练开始了")
    epoch_start_time = time.time()  # 记录 epoch 开始时间

    # 训练开始
    jl.train()  # 切换到训练模式
    for data in train_dataloader:
        imgs, targets = data
        imgs, targets = imgs.to(device), targets.to(device)  # 将数据移动到指定设备

        outputs = jl(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_train_step += 1

        if total_train_step % 100 == 0:
            print(f"训练次数 {total_train_step}, Loss: {loss.item()}")

            epoch_end_time = time.time()  # 记录 epoch 结束时间
            print(f"耗时: {epoch_end_time - epoch_start_time:.2f} 秒")

    # 测试过程
    jl.eval()  # 切换到评估模式
    with torch.no_grad():
        total_test_loss = 0
        for data in test_dataloader:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)  # 将数据移动到指定设备

            outputs = jl(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()

    print(f"整体测试集上的 Loss: {total_test_loss:.4f}")

    # 保存模型
    torch.save(jl.state_dict(), f"jl_{i}.pth")
    print("模型已经保存")
