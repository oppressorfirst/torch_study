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
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# 创建一个神经网络实例，并将其移动到设备上，并打印其结构。
model = NeuralNetwork().to(device)
print(model)

# 为了使用模型，我们会将输入数据传递给它。这将执行模型的forward，以及一些后台操作。不要直接调用 model.forward()！
# 在输入上调用模型会返回一个二维张量（在使用batch的时候会体现），其中 dim=0 对应于每个类别的 10 个原始预测值，dim=1 对应于每个输出的单个的值。
# 通过将其传递给 nn.Softmax 模块的实例，我们可以得到预测概率。
X = torch.rand(2, 28, 28, device=device)
logits = model(X)
print(logits)
pred_probab = nn.Softmax(dim=1)(logits)
print(pred_probab)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

# #Model中的层
# 来分解一下FashionMNIST模型中的层级。
# 为了说明这一点，我们将取一个包含3张图片的大小为28x28的样本mini-batch，并看看当我们将其通过网络时会发生什么。
input_image = torch.rand(3, 28, 28)
print(input_image.size())

# nn.Flatten层
# 我们初始化一个nn.Flatten层将每个2D的28x28图片转换为一个连续的包含784个像素值的数组
# （批量的这个维度（在dim=0处，形容有三个图片的维度）保持不变）
flatten = nn.Flatten()
flat_image = flatten(input_image)
print("flat_image: ", flat_image.size())

# nn.Linear层
# nn.Linear层是一个torch网络模块，它使用其存储的权重和偏置对input images进行线性变换。
layer1 = nn.Linear(in_features=28 * 28, out_features=20)
hidden1 = layer1(flat_image)  # 最后提取出来20个元素
print("hidden1.size():", hidden1.size())

# nn.ReLU层
# 非线性激活函数是创建模型输入和输出之间复杂映射的关键。
# 它们在线性变换之后应用，引入非线性，有助于神经网络学习各种现象。
# 在这个模型中，我们在线性层之间使用nn.ReLU，但是还有其他激活函数可以用来在模型中引入非线性。
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# nn.Sequential容器
# nn.Sequential是一个ordered容器模块。
# 数据通过所有模块的顺序与定义的顺序相同。
# 您可以使用nn.Sequential容器来组合一个类似`seq_modules`的快速网络。
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)

# nn.Softmax层
# 神经网络的最后一个线性层返回logits。
# nn.Softmax层接收[-infty, infty]范围内的原始值，并且将logits缩放到[0, 1]范围中。
# 缩放之后的数字表示每个类别的模型预测概率。dim参数指示值必须在这个维度上求和得到1。
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)


# # 模型参数
# 神经网络中的许多层都是参数化的，即具有相关联的权重和偏置，在训练期间进行优化。
# 子类化nn.Module会自动跟踪模型对象中定义的所有字段，并使所有参数可通过模型的parameters()或named_parameters()方法访问。
# 在这个例子中，我们遍历每个参数，并打印其大小和值的预览。
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
