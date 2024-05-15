import os

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import DataLoader

# #处理数据样本的代码经常会变得臃肿和难以维护，所以想改变一下
# pytorch提供了两个原型让你可以预读取你的数据
# torch.utils.data.Dataset保存了将要使用的数据和他们的标签
# torch.utils.data.DataLoader将Dataset中的数据包装成为可迭代对象，方便访问
# torch还提供了一系列的数据集可以作为原型

# #加载一个数据集
# 加载Fashion_MNIST数据集
"""
:param root: 数据集存储的位置
:param train: 标记是训练集还是测试集
:param transform 和 target_transform指定了feature和标签的一系列变换
"""
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# #数据集的迭代和可视化
# 我们可以直接使用training_data[index]来访问数据集中的数据。
# 也可以使用matplotlib来让一些数据可视化
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
sample_idx = torch.randint(len(training_data), size=(1,)).item()
img, label = training_data[sample_idx]
plt.title(labels_map[label])
plt.axis("off")
plt.imshow(img.squeeze(), cmap="gray")
plt.show()


# #创造你自己的数据集
# 创造一个自己的数据集必须实现下面三个函数：1.__init__() 2.__getitem__() 3.__len__()
# 图片存放在img_dir， 标签信息存放在labels_file中
class MyDataset(Dataset):
    # 实例化Dataset对象时，__init__会运行一次。我们初始化包含图像、注释文件和两种转换的一个空间。
    def __init__(self, img_dir, labels_file, transform=None, target_transform=None):
        # 先把图片的标签指定了
        self.img_labels = pd.read_csv(labels_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    # 返回我们所有数据集中所有数据的长度
    def __len__(self):
        return len(self.img_labels)

    # __getitem__函数从给定的索引idx处的加载数据集中的样本并返回
    # 基于该索引idx，函数识别图像在磁盘上的位置，使用read_image将其转换为tensor，
    # 从self.img_labels中的CSV数据中检索相应的label，调用它们的transform（如果适用）
    # 最终返回tensor image和对应的标签(以元组的形式)。
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        image_label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            image_label = self.target_transform(image_label)

        return image, image_label


# # 使用DataLoaders来准备训练用的数据
# Dataset一次只能加载数据集中的一个样本，但是在训练模型的时候，我们会使用mini-batches的方法，
# 还会使用不同的打乱顺序来降低过拟合。并且使用Python的multiprocessing来加快数据检索。
# DataLoader是一个迭代器，它在一个简单的API中为我们抽象了这种复杂性。

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# 我们已经将该数据集加载到 DataLoader 中，可以根据需要迭代数据集。
# 下面的每次迭代都会返回一批 train_features 和 train_labels（分别包含 batch_size=64 的特征和标签）。
# 因为我们指定了 shuffle=True，所以在迭代完所有批次后，数据会被打乱

# 展示图片和标签
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()
print(f"Label: {label}")


