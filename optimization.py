# #优化模型的参数
# 现在我们有了一个模型和数据，是时候在数据上训练、验证和测试我们的模型，最终优化我们的模型参数了。
# 训练模型是一个迭代过程；在每次迭代中，模型对输出进行猜测，计算其猜测的错误（损失），
# 收集相对于其参数的错误的导数（如我们在前面部分所见），并使用梯度下降优化这些参数。

# 前置准备的代码
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

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

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


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


model = NeuralNetwork()

# 超参数
# 超参数是可调参数，它们可以让您控制模型优化过程。
# 不同的超参数值可以影响模型训练和收敛速度（了解更多关于超参数调优的信息）。
learning_rate = 1e-3
batch_size = 64
epochs = 5

# # 优化循环
# 一旦我们设置了超参数，我们就可以使用优化循环来训练和优化我们的模型。优化循环的每次迭代称为一个epoch。
# 每个epoch包括两个主要部分：
# - 训练循环：遍历训练数据集并尝试收敛到最佳参数。
# - 验证/测试循环：遍历测试数据集以检查模型性能是否在改善。

# #损失函数
# 当提供一些训练数据时，我们的未经训练的网络很可能无法给出正确的答案。
# 损失函数度量了获得的结果与目标值之间的不相似程度，而在训练过程中我们希望最小化的正是这个损失函数。
# 为了计算损失，我们使用给定数据样本的输入进行预测，并将其与真实的数据标签值进行比较。
# 常见的损失函数包括用于回归任务的nn.MSELoss（均方误差），以及用于分类任务的nn.NLLLoss（负对数似然）。
# nn.CrossEntropyLoss将nn.LogSoftmax和nn.NLLLoss组合在一起。
# 我们将模型的输出logits传递给nn.CrossEntropyLoss，它将对logits进行归一化并计算预测误差。
loss_fn = nn.CrossEntropyLoss()

# #优化器
# 优化是调整模型参数以在每个训练步骤中减少模型错误的过程。
# 优化算法定义了此过程如何执行（在本例中，我们使用随机梯度下降）。
# 所有优化逻辑都封装在优化器对象中。
# 在这里，我们使用SGD优化器；此外，PyTorch中还有许多不同的优化器，如ADAM和RMSProp，适用于不同类型的模型和数据。
# 我们通过注册需要训练的模型参数并传入学习率超参数来初始化优化器。
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# 在训练循环内部，优化过程分为三个步骤：
# 可以先计算模型的输出，再计算loss函数的新值
# 调用optimizer.zero_grad()来重置模型参数的梯度。梯度默认会累加；为了防止重复计数，我们在每次迭代时显式地将它们归零。
# 使用loss.backward()进行预测损失的反向传播。PyTorch会将损失相对于每个参数的梯度存储起来。
# 一旦获得了梯度，我们就调用optimizer.step()来根据反向传播中收集到的梯度来调整参数。

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
