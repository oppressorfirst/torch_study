import torch

# 检查 MPS 是否可用
if torch.backends.mps.is_available():
    device = torch.device("mps")  # 设置设备为 MPS
    print("Using MPS for acceleration")
else:
    device = torch.device("cpu")
    print("MPS not available, using CPU")
