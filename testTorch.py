import torch

# Pytorchのバージョン
print(torch.__version__)

# GPUが利用可能であるか否か
print(torch.cuda.is_available())

# 利用可能なGPUの名前
print(torch.cuda.get_device_name(0))  
