from ultralytics import YOLO

# 加载模型
model = YOLO('runs/detect/train46_epoch133_pretrain_autoconverge/weights/best.pt')  # 预训练的 YOLOv8n 模型
# model = YOLO('yolov8n.pt')  # 预训练的 YOLOv8n 模型

# 在图片列表上运行批量推理
# results = model(['D:\\ultralytics-main\\ultralytics\\assets\\zidane.jpg'], stream=True)  # 返回 Results 对象生成器
model.predict('C:\\Users\\Administrator\\Desktop\\Dataset\\images\\trainval72.jpg')
