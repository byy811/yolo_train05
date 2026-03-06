from ultralytics import YOLO

# 加载官方预训练模型
model = YOLO('../yolov8n.pt')

# 随便找张图预测一下（它会自动下载模型）
results = model('https://ultralytics.com/images/bus.jpg')
print("环境搭建成功！")