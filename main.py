from ultralytics import YOLO

if __name__ == '__main__':
    # 加载一个模型
    model = YOLO('yolov8l.yaml')  # 从YAML建立一个新模型
    model = YOLO('yolov8l.pt')  # 加载预训练模型（推荐用于训练）
    model = YOLO('yolov8l.yaml').load('yolov8l.pt')  # 从YAML建立并转移权重

    # 训练模型
    results = model.train(data='D:\\ultralytics-main\\YOLO_dataset\\mydata.yaml', epochs=200, imgsz=640,batch=8,workers=1)

##数据集问题
'''
数据集问题很大:   
1.多人图片对目标检测的影响：多张人脸，一个label，造成多人脸图像容易只识别出一张脸(解决方案：随机裁剪（数据增强）可缓解一部分)
2.多人图片对年龄回归影响：一旦检测出多张人脸，人脸和标签极其容易不对应（解决方案：写了一个get_closest_box函数，取出与标签最近的人脸）
3.多人图片中很多人脸标签就直接是错的，人脸和年龄对应不上（如test1，test2）
4.不同光照条件下的人脸影响年龄判断，加入对比度数据增强效果明显提升
4.小孩脸数据集太少，导致小孩脸的检测效果差，多人图像中检测不出来
（试试数据增强）
5.可以试试分类网络（分年龄段（大致分5类））级联回归网络

'''