from ultralytics import YOLO

# 如果这一行不报错，且能打印出模型结构，就说明成功了！
model = YOLO("yolov8n.pt")
print("环境配置成功！")