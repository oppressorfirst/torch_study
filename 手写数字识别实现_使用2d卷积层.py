import torch
import torch.nn as nn
import torchvision.transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pathlib import Path

# 检查是否有可用的GPU
device = "cuda"
batch_size = 64
dataset_dir = Path("dataset")
train_img_dir = dataset_dir / "MNIST" / "train"
test_img_dir = dataset_dir / "MNIST" / "test"

transform = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(),  # 将图像转换为灰度图
    torchvision.transforms.ToTensor(),  # 将图像转换为张量
])

target_transform = torchvision.transforms.Compose([
    torchvision.transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
])

train_Dataset = ImageFolder(root=str(train_img_dir), transform=transform, target_transform=target_transform)
test_Dataset = ImageFolder(root=str(test_img_dir), transform=transform, target_transform=target_transform)

# 将数据加载到GPU上执行
train_DataLoader = DataLoader(dataset=train_Dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_DataLoader = DataLoader(dataset=test_Dataset, batch_size=batch_size, shuffle=True, drop_last=True)


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1,16,4,stride=1),
            nn.ReLU(),
            nn.Conv2d(16,64,16,stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 8, 4, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7*7*8, 10),
        )

    def forward(self, x):
        res = self.seq(x)
        return res


# 实例化模型并将其移到GPU上执行
model = MyModel().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train_loop(DataLoader, model, loss_fn, optimizer):
    size = len(DataLoader.dataset)
    model.train()
    for batch, (X, y) in enumerate(DataLoader):
        # 将数据移到GPU上执行
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(DataLoader, model, loss_fn):
    size = len(DataLoader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in DataLoader:
            # 将数据移到GPU上执行
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= batch_size
    correct /= size
    torch.save(model, '手写数字卷积层模型2.pth')
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for t in range(10000):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_DataLoader, model, loss_fn, optimizer)
    test_loop(test_DataLoader, model, loss_fn)
print("Done!")
