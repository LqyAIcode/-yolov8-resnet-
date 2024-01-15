from ultralytics import YOLO

# 加载预训练的YOLOv8n模型
model = YOLO('yolov8n.pt')

# 在图片上运行推理
results = model('D:\\ultralytics-main\\ultralytics\\assets\\bus.jpg')

# 查看结果
for r in results:
    print(r.boxes.xyxy)  # 打印包含检测边界框的Boxes对象