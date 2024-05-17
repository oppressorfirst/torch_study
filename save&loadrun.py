# #保存和加载训练好的模型
# 在本节中，我们将了解如何通过保存、加载和运行模型预测来保持模型状态。
import torch
import torchvision.models as models

# #保存和加载模型的权重
# PyTorch 模型将学习到的参数存储在一个内部状态字典中，称为 state_dict。可以通过 torch.save 方法将其保存。
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')

# 为了加载模型的参数，需要现将相同的模型实例化，再通过load_state_dict()方法加载模型
model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
# 注意：
# 在推理之前，一定要调用 model.eval() 方法，以将 dropout 和 batch normalization 层设置为评估模式。
# 如果不这样做，会导致推理结果不一致。

# 将整个模型都全部保存
# 在加载模型权重时，我们需要先实例化模型类，因为类定义了网络的结构。
# 我们可能希望将这个类的结构与模型一起保存，在这种情况下，可以将模型（而不是 model.state_dict()）传递给保存函数：
torch.save(model, '手写数字线性层模型.pth.pth')

# 可以使用下面的方法加载模型
model = torch.load('手写数字线性层模型.pth.pth')

# 注意：
# 这种方法在序列化模型时使用了 Python 的 pickle 模块，因此在加载模型时，需要确保实际的类定义是可用的。
