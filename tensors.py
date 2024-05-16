import torch
import numpy as np

# # 创建tensor
# 从数据中直接创建tensors
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# 从ndarray直接变化成tensors
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# 从另一个tensor变换(新tensor保留参数张量的属性(形状，数据类型)，除非显式覆盖。)
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)     # overrides the datatype of x_data,形状没有改变
print(f"Random Tensor: \n {x_rand} \n")

# 从随机的或者常数转化成tensors：
# shape 是张量维的元组。在下面的函数中，它决定了输出张量的维数。
shape = (4,5,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# # tensor的属性
# tensor 属性描述它们的形状、数据类型和存储它们的设备。
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# # tensors上的操作
# 在pytorch上，我们可以对tensors在GPU上进行多达100多种操作，例如矩阵转置，矩阵采样，矩阵切片等等
# 首先验证GPU是否可用
if torch.cuda.is_available():
    print(torch.cuda.is_available())
    tensor = tensor.to("cuda")

# 基础的类似numpy的取值和切片
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0     # 让第二列全部置零
print(tensor)

# 连接多个tensors 你可以用torch.cat。沿着给定的维数串联tensors序列
t1 = torch.cat([tensor, tensor, tensor], dim=0)
print("t1:",t1)

# #算术运算
# 这个算的是矩阵惩乘法. y1, y2, y3 有相同的输出
# ``tensor.T`` 返回的是tensor的转置
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)
print(y1)

# 这个算的是元素乘积。 z1, z2, z3有相同的输出
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(z1)

# #单元素的tensors
# 如果有一个单元素tensor，可以使用item()将其转换为Python数值
# 可以使用tensor.sum()来将tensor中所有值加在一起
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# #就地操作 即时改变操作将结果存储到操作数中。它们用后缀_表示。例如:x.copy_(y)， x.t_()，将改变x
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

# #和numpy之间的关系
# tensor到ndarray
t = torch.ones(5)
print(f"t: {t}, type of n: {type(t)}")
n = t.numpy()
print(f"n: {n}, type of n: {type(n)}")

# 在tensor中的改变会反映到ndarray中
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# ndarray 转化到tensor
n = np.ones(5)
t = torch.from_numpy(n)

# 在ndarray中的改变会反映到tensor中
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")



