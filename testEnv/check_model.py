"""
验证模型配置是否正确
"""

from ultralytics import YOLO
import torch

print("="*60)
print("模型结构验证")
print("="*60)

# 1. 测试 CBAM 版本
print("\n1. 检查 YOLOv8n-CBAM 模型...")
try:
    model = YOLO('yolov8-cbam.yaml')
    model.load('yolov8n.pt')  # 加载预训练权重

    # 打印模型信息
    model.info(detailed=False)

    # 查找 CBAM 层
    cbam_found = False
    for name, module in model.model.named_modules():
        if 'CBAM' in str(type(module).__name__):
            print(f"\n✅ 找到 CBAM 层: {name}")
            print(f"   类型: {type(module)}")
            cbam_found = True

    if not cbam_found:
        print("\n❌ 未找到 CBAM 层！")

    # 测试前向传播
    print("\n2. 测试前向传播...")
    dummy_input = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        output = model(dummy_input)
    print("✅ 前向传播成功！")

except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)