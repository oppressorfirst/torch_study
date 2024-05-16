# # 使用torch.autograd自动计算微分
# 在训练神经网络时，最常用的算法是反向传播。在这个算法中，参数（模型权重）根据损失函数相对于给定参数的梯度进行调整。
# 为了计算这些梯度，PyTorch拥有一个内置的微分引擎叫做torch.autograd。它支持对任何计算图的梯度进行自动计算。
# 考虑到最简单的单层神经网络，包含输入x、参数w和b，以及一些损失函数。可以使用以下方式在PyTorch中定义它：
import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
# 实现了这个算法图：comp-graph.png
# 在这个网络中，w和b是需要优化的参数。
# 因此，我们需要能够计算损失函数相对于这些变量的梯度。为了实现这一点，我们设置这些张量的requires_grad属性是需要梯度的。
# 注意：您可以在创建张量时设置requires_grad的值，也可以稍后使用x.requires_grad_(True)方法进行设置。
print(loss)

# tensor上应用的构建计算图的函数实际上是一个Function类的对象。
# 该对象知道如何在正向方向计算函数，并且在反向传播步骤中也知道如何计算其导数。
# 反向传播函数的引用存储在张量的grad_fn属性中。
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# # 计算梯度
# 为了优化神经网络中的参数权重，我们需要计算损失函数相对于参数的导数，
# 即我们需要在一些固定的x和y值下计算∂loss\∂w和∂loss\∂b。
# 为了计算这些导数，我们调用loss.backward()，然后从w.grad和b.grad中检索值：
loss.backward()
# 要先对loss函数进行反向传播之后才能计算参数的梯度
print(w.grad)
print(b.grad)
# 我们只能获取计算图的requires_grad属性设置为True的最终节点的grad属性。对于图中的所有其他节点，梯度将不可用。
# 出于性能原因，我们只能在给定图上执行一次反向传播的梯度计算。
# 如果我们需要在同一图上进行多次反向传播调用，则需要在backward调用中传递retain_graph=True。

# #关闭梯度追踪
# 默认情况下，所有requires_grad=True的张量都在跟踪其计算历史并支持梯度计算。
# 然而，在某些情况下，我们并不需要这样做，
# 例如，当我们已经训练好模型，只想将其应用于一些输入数据时，即我们只想通过网络进行前向计算。
# 我们可以通过将我们的计算代码放入torch.no_grad()块中来停止跟踪计算。
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

# 实现相同效果的另一种方法是使用Tensor上的detach()方法：
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)


