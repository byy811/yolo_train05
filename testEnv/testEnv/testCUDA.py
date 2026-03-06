import sys
print(f"Python 版本: {sys.version}")
print(f"Python 位数: {sys.maxsize > 2**32 and '64位' or '32位'}")

try:
    import torch
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
except Exception as e:
    print(f"PyTorch 加载失败: {e}")