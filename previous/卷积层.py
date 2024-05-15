import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=False)
dataloader = DataLoader(dataset, batch_size=64)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = Conv2d(3, 6, 3, 1, 0)
        # self.conv2 = Conv2d(3, 8, 4, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        return x


cnn = CNN()


for data in dataloader:
    imgs, targets = data
    output = cnn(imgs)
    print(output.shape)

