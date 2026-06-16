# 导入PyTorch核心库
import torch
# 导入神经网络模块（nn = Neural Network），包含所有预定义的层和损失函数
import torch.nn as nn
# 导入优化器模块，包含各种梯度下降算法
import torch.optim as optim
# 导入数据处理和加载工具
from torch.utils.data import DataLoader
# 导入计算机视觉相关的数据集和变换
from torchvision import datasets, transforms

# 导入numpy用于数据处理
import numpy as np
# 导入matplotlib用于可视化
import matplotlib.pyplot as plt
# 打印PyTorch版本
print(f"PyTorch版本: {torch.__version__}")

# 检查CUDA是否可用（CUDA是NVIDIA提供的GPU加速计算平台）
cuda_available = torch.cuda.is_available()
print(f"CUDA是否可用: {cuda_available}")

if cuda_available:
    # 打印GPU数量
    print(f"GPU数量: {torch.cuda.device_count()}")
    # 打印当前使用的GPU名称
    print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    # 设置默认设备为GPU
    device = torch.device("cuda")
else:
    # 如果没有GPU，使用CPU
    device = torch.device("cpu")

print(f"使用的设备: {device}")


# 1. 从Python列表创建张量
tensor_from_list = torch.tensor([1, 2, 3, 4])
print("从列表创建的张量:", tensor_from_list)

# 2. 创建全0张量，参数是形状(shape)
zeros_tensor = torch.zeros((2, 3))  # 2行3列的全0矩阵
print("\n全0张量:\n", zeros_tensor)

# 3. 创建全1张量
ones_tensor = torch.ones((3, 2))  # 3行2列的全1矩阵
print("\n全1张量:\n", ones_tensor)

# 4. 创建指定范围内的连续整数张量
range_tensor = torch.arange(0, 10, 2)  # 从0到10（不包含10），步长为2
print("\n范围张量:", range_tensor)

# 5. 创建随机张量（值在0到1之间均匀分布）
rand_tensor = torch.rand((2, 2))
print("\n随机张量:\n", rand_tensor)

# 6. 创建随机张量（值服从标准正态分布，均值为0，方差为1）
randn_tensor = torch.randn((3, 3))
print("\n正态分布随机张量:\n", randn_tensor)

# 创建一个示例张量
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

# 1. 查看张量的形状（shape）
print("张量形状:", x.shape)
print("张量形状（另一种写法）:", x.size())  # 与shape等价

# 2. 查看张量的维度数
print("张量维度数:", x.ndim)

# 3. 查看张量的数据类型（dtype）
print("张量数据类型:", x.dtype)

# 4. 转换张量的数据类型
x_int = x.to(torch.int32)
print("\n转换为int32后的张量:\n", x_int)
print("转换后的数据类型:", x_int.dtype)

# 创建两个示例张量
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

# 1. 逐元素运算（element-wise）
print("逐元素加法:\n", a + b)
print("逐元素减法:\n", a - b)
print("逐元素乘法:\n", a * b)
print("逐元素除法:\n", a / b)

# 2. 矩阵乘法（这是神经网络中最重要的运算！）
# 线性层的核心就是矩阵乘法
matrix_mul = torch.matmul(a, b)
print("\n矩阵乘法结果:\n", matrix_mul)
# 简写形式
matrix_mul_short = a @ b
print("矩阵乘法简写结果:\n", matrix_mul_short)

# 3. 张量的聚合运算
print("\n张量所有元素的和:", a.sum())
print("张量所有元素的平均值:", a.mean())
print("张量的最大值:", a.max())
print("张量的最小值:", a.min())

# 创建一个示例张量
x = torch.arange(1, 13)  # 1维张量，包含1到12的整数
print("原始张量:", x)
print("原始形状:", x.shape)

# 1. reshape：改变张量的形状，元素总数不变
x_reshaped = x.reshape(3, 4)  # 变成3行4列的2维张量
print("\nreshape为(3,4):\n", x_reshaped)
print("reshape后的形状:", x_reshaped.shape)

# 2. transpose：交换两个维度
x_transposed = x_reshaped.transpose(0, 1)  # 交换第0维和第1维
print("\ntranspose后:\n", x_transposed)
print("transpose后的形状:", x_transposed.shape)

# 3. flatten：将多维张量展平为1维张量
# 这在卷积层连接到线性层时非常常用
x_flattened = x_reshaped.flatten()
print("\nflatten后:", x_flattened)
print("flatten后的形状:", x_flattened.shape)

# 4. unsqueeze：增加一个维度
x_unsqueezed = x.unsqueeze(0)  # 在第0维增加一个维度
print("\nunsqueeze后形状:", x_unsqueezed.shape)  # 从(12,)变成(1, 12)


# 检查我们之前定义的device变量
print(f"当前使用的设备: {device}")

# 创建一个CPU上的张量
x_cpu = torch.tensor([1, 2, 3])
print("\n张量所在设备:", x_cpu.device)

# 1. 将张量移动到GPU
x_gpu = x_cpu.to(device)
print("移动到GPU后的设备:", x_gpu.device)

# 2. 直接在GPU上创建张量
x_gpu_direct = torch.tensor([4, 5, 6], device=device)
print("直接在GPU上创建的张量设备:", x_gpu_direct.device)

# 3. 将GPU上的张量移回CPU
x_cpu_back = x_gpu.cpu()
print("移回CPU后的设备:", x_cpu_back.device)

# 4. 重要：模型和数据必须在同一个设备上才能运算！
# 错误示例：CPU张量和GPU张量不能直接相加
try:
    result = x_cpu + x_gpu
except RuntimeError as e:
    print("\n错误示例（CPU和GPU张量相加）:", e)

# 正确示例：都移到同一个设备
result = x_cpu.to(device) + x_gpu
print("正确运算结果:", result)


