"""
测试 Ultralytics 是否正确安装为可编辑模式
"""

import ultralytics
from ultralytics import YOLO

print("="*60)
print("Ultralytics 源码路径测试")
print("="*60)
print(f"Ultralytics 安装路径: {ultralytics.__file__}")
print(f"Ultralytics 版本: {ultralytics.__version__}")

# 测试是否能正常加载模型
try:
    model = YOLO('yolov8n.pt')
    print("\n✅ 模型加载成功！")
    model.info()
    print("\n✅ 所有测试通过！可以开始修改源码了！")
except Exception as e:
    print(f"\n❌ 测试失败: {e}")