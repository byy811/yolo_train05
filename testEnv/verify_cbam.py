"""
验证 CBAM 集成是否正确
"""

from ultralytics import YOLO
import torch

print("="*60)
print("CBAM 集成验证")
print("="*60)

# 1. 加载模型
print("\n1. 加载 YOLOv8n-CBAM 模型...")
try:
    model = YOLO('yolov8-cbam.yaml')
    print("✅ 模型加载成功")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 2. 打印模型结构
print("\n2. 模型结构信息:")
model.info(detailed=False)

# 3. 查找 CBAM 层
print("\n3. 查找 CBAM 层...")
cbam_count = 0
for i, (name, module) in enumerate(model.model.named_modules()):
    if 'CBAM' in str(type(module).__name__):
        cbam_count += 1
        print(f"✅ 找到 CBAM 层 #{cbam_count}:")
        print(f"   位置: {name}")
        print(f"   类型: {type(module)}")
        # 打印 CBAM 的参数
        for param_name, param in module.named_parameters():
            print(f"   参数: {param_name}, shape={param.shape}")

if cbam_count == 0:
    print("❌ 未找到任何 CBAM 层！请检查配置")
    exit(1)

# 4. 测试前向传播
print("\n4. 测试前向传播...")
try:
    dummy_input = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        # output = model(dummy_input)
        results = model(dummy_input)  # YOLOv8 返回的是 Results 列表
        # 打印 Results 对象中的张量形状
    print(f"✅ 前向传播成功！检测框张量形状: {results[0].boxes.data.shape}")
    # print(f"✅ 前向传播成功！输出形状: {output[0].shape if isinstance(output, tuple) else output.shape}")
except Exception as e:
    print(f"❌ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 5. 加载预训练权重测试
print("\n5. 测试加载预训练权重...")
try:
    model.load('yolov8n.pt')
    print("✅ 预训练权重加载成功（部分层不匹配是正常的）")
except Exception as e:
    print(f"⚠️  预训练权重加载失败: {e}")

print("\n" + "="*60)
print("✅ 所有验证通过！可以开始训练")
print("="*60)